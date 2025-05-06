import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import cosine_beta_schedule, default, extract
from .UNet import UNet
# from .UViT import UViT
# from SiamB.core.defaults import cfg
import matplotlib.pyplot as plt
import gc
def get_l2(f):
    c, h, w = f.shape
    mean = torch.mean(f, dim=0).unsqueeze(0).repeat(c, 1, 1)
    f = torch.sum(torch.pow((f - mean), 2), dim=0) / c
    f = f
    return (f - f.min()) / (f.max() - f.min())

def draw(f, name):
    plt.imshow(get_l2(f).detach().cpu().numpy())
    plt.title(name)
    plt.show()

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim).cuda() * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Time_Embedding(nn.Module):
    def __init__(self, cfg):
        super(Time_Embedding, self).__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(int(cfg.Time_Embedding.Dim / 4)),
            nn.Linear(int(cfg.Time_Embedding.Dim / 4), cfg.Time_Embedding.Dim),
            nn.GELU(),
            nn.Linear(cfg.Time_Embedding.Dim, cfg.Time_Embedding.Dim),
        )
    def forward(self, t):
        return self.time_mlp(t)

class LDDiffuse(nn.Module):
    def __init__(self, Total=1000, Sample=5, image_size=16):
        super(LDDiffuse, self).__init__()
        # define a track head
        # self.time_embedding = Time_Embedding(cfg)
        ## set T
        self.total_timesteps = Total
        sampling_timesteps = Sample
        self.objective = 'pred_x0'
        ##prepare beta and alpha
        betas = cosine_beta_schedule(self.total_timesteps)                             # beta
        alphas = 1. - betas                                                            # 1-beta = alpha
        alphas_cumprod = torch.cumprod(alphas, dim=0)                                  # (1-beta)! = alpha! = alpha bar
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.condition = True
        # self.scale = cfg.Diffusion.SNR_SCALE
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        self.num_timesteps = Total
        self.sample_timesteps = Sample
        self.set_loss('l2')

        self.unet = UNet(conditional=True, image_size=image_size)
        # self.unet = UViT(conditional=True)

    def set_loss(self, type):
        if type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod.cpu(), t.cpu(), x_t.shape) * x_t.cpu() - x0.cpu()) /
                extract(self.sqrt_recipm1_alphas_cumprod.cpu(), t.cpu(), x_t.shape)
        ).float()

    # forward diffusion
    def q_sample(self, x_start, noise_distractor, t, noise=None):
        a = 0.5
        if noise is None:
            noise = torch.randn_like(x_start)
        # noise = (1 - a) * noise + a * noise_distractor
        # mean = noise.mean()
        # std = noise.std()
        # noise = (noise - mean) / (std + 1e-8)
        if np.random.rand() < 0.5:
            noise = (1-a) * noise + a * noise_distractor
            # noise = noise_distractor
            mean = noise.mean()
            std = noise.std()
            noise = (noise - mean) / (std + 1e-8)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return (sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise).float(), sqrt_alphas_cumprod_t.float(), sqrt_one_minus_alphas_cumprod_t.float()

    # backward generation
    def p_sample(self, time, time_next, before, pred_x0, pred_noises):
        if time_next == -1:
            return before
        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]

        sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(before)
        # x_start = pred[-1][:, 1, :, :]

        # pred_noise = self.predict_noise_from_start(before, time, pred[-1])
        # predict_x0 = self.get_predict_x0(before, pred, [beta], [one_beta])
        out = alpha_next.sqrt() * pred_x0 + sigma * noise + c * pred_noises
        # xt_next = alpha_next.sqrt() * pred +
        # out = pred[-1] * alpha_next.sqrt() + \
        #       c * pred_noise + \
        #       sigma * noise
        return out

    ## get the random t and noise for each sample within the batch
    def prepare_targets(self, num, template):
        ts = []
        noises = []
        for i in range(num):
            t = torch.randint(0, self.num_timesteps, (1,)).long().cuda()
            noise = torch.randn_like(template).cuda()
            ts.append(t)
            noises.append(noise)
        return ts, noises

    def get_predict_x0(self, xt, noises, betas):
        x0 = [(xt[i] - (1-betas[i]).sqrt() * noises[i]) / betas[i].sqrt() for i in range(len(betas))]
        return x0

    def predict(self, fea_rgb, fea_tir, GAT_level1):
        # get time steps
        times = torch.linspace(-1, self.num_timesteps - 1, self.sample_timesteps + 1).cuda()
        times = list(reversed(times.int().tolist()))
        # times = [999,2,-1]
        time_pairs = list(zip(times[:-1], times[1:]))
        # time_pairs = time_pairs[1:]
        ol = []
        pl = []
        xl = []
        xt = torch.randn_like(fea_rgb).cuda()
        for time, time_next in time_pairs:
            # time_cond = torch.full((1,), time, dtype=torch.long).cuda()
            # time = 500
            xl.append(xt)
            time = torch.full((1,), time, dtype=torch.long).cuda()
            beta = extract(self.alphas_cumprod, time, xt.shape).float()
            # one_beta = extract(self.sqrt_one_minus_alphas_cumprod, time, xt.shape).float()
            tem = torch.cat((torch.cat((xt.cpu(), fea_rgb.cpu()), dim=1), fea_tir.cpu()), dim=1)
            out = self.unet(tem, beta, fea_rgb, GAT_level1)
            ol.append(out)
            pred_n = self.predict_noise_from_start(xt, time, out)
            pl.append(pred_n)
            # predict_x0 = self.get_predict_x0(xt, out_noises, [beta])[-1]
            # xt = self.p_sample(time, time_next, xt, predict_x0, out_noises).float()
            if time_next < 0:
                # img = out_cls[:,1,:,:]
                return out
            else:
                xt = self.p_sample(time, time_next, xt, out, pred_n).float()
                # del out, pred_n
                # del out, pred_n
                # gc.collect()
            # att = self.diffusion_model.p_sample(time_cond, time_next, att, out_cls)

    def get_predicts(self, qs, ts, noises, betas):
        # out = []
        # for i in range(qs.size(0)):
        out = self.get_predict_x0(qs, noises, betas)
        # out = torch.stack(out)
        out = [self.p_sample(ts[i], ts[i]-1, qs[i], out[i], noises[i]).float() for i in range(betas.size(0))]
        return torch.stack(out)

    def write_times(self, times):
        s = ''
        for time in times:
            s = s + str(time.item()) + ' '
        s = s + '\n'


    def forward(self, fused_fea, last_fusedfea_asnoise, GAT_level1, fea_rgb, fea_tir):
        sample_num = fused_fea.size(0)
        # prepare
        ts, noises = self.prepare_targets(sample_num, fused_fea[0])
        self.write_times(ts)
        qs = []
        betas = []
        one_betas = []
        for i in range(sample_num):
            q, beta, one_beta = self.q_sample(fused_fea[i], last_fusedfea_asnoise[i], ts[i], noises[i])
            qs.append(q)
            betas.append(beta)
            one_betas.append(one_beta)
        if not self.condition:
            out = self.unet(qs, betas)
        else:
            qs = torch.stack(qs)
            betas = torch.stack(betas)
            con = torch.cat((fea_rgb, fea_tir), dim=1)
            out = self.unet(torch.cat((qs, con), dim=1), betas, fea_rgb, GAT_level1)
        # self.loss_func.cuda()
        noises = torch.stack(noises)
        pred_n = [self.predict_noise_from_start(qs[i], ts[i], out[i]) for i in range(len(ts))]
        pred_n = torch.stack(pred_n)
        loss = self.loss_func(noises, pred_n).cuda()
        # predicted_x0s = self.get_predicts(qs, ts, out_noises, betas)
        return loss, out




