# import math
# import torch
# from torch import nn
# import torch.nn.functional as F
# from inspect import isfunction
# import matplotlib.pyplot as plt
# import gc
# def get_l2(f):
#     c, h, w = f.shape
#     mean = torch.mean(f, dim=0).unsqueeze(0).repeat(c, 1, 1)
#     f = torch.sum(torch.pow((f - mean), 2), dim=0) / c
#     f = f
#     return (f - f.min()) / (f.max() - f.min())
#
# def draw(f, name):
#     plt.imshow(get_l2(f).detach().cpu().numpy())
#     plt.title(name)
#     plt.show()
#
# def exists(x):
#     return x is not None
#
#
# def default(val, d):
#     if exists(val):
#         return val
#     return d() if isfunction(d) else d
#
# # PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
# class PositionalEncoding(nn.Module):
#     def __init__(self, dim):
#         super(PositionalEncoding, self).__init__()
#         self.dim = dim
#
#     def forward(self, noise_level):
#         count = self.dim // 2
#         step = torch.arange(count, dtype=noise_level.dtype,
#                             device=noise_level.device) / count
#         encoding = noise_level.unsqueeze(
#             1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
#         encoding = torch.cat(
#             [torch.sin(encoding), torch.cos(encoding)], dim=-1)
#         return encoding
#
#
# class FeatureWiseAffine(nn.Module):
#     def __init__(self, in_channels, out_channels, use_affine_level=False):
#         super(FeatureWiseAffine, self).__init__()
#         self.use_affine_level = use_affine_level
#         self.noise_func = nn.Sequential(
#             nn.Linear(in_channels, int(out_channels*(1+self.use_affine_level)))
#         )
#
#     def forward(self, x, noise_embed):
#         batch = x.shape[0]
#         if self.use_affine_level:
#             gamma, beta = self.noise_func(noise_embed).view(
#                 batch, -1, 1, 1).chunk(2, dim=1)
#             x = (1 + gamma) * x + beta
#         else:
#             x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
#         return x
#
#
# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)
#
#
# class Upsample(nn.Module):
#     def __init__(self, dim, new_res):
#         super(Upsample, self).__init__()
#         # self.up = nn.Upsample(size=new_res, scale_factor=2, mode="nearest")
#         self.up = nn.Upsample(size=new_res, mode="nearest")
#         self.conv = nn.Conv2d(dim, dim, 3, padding=1)
#
#     def forward(self, x):
#         return self.conv(self.up(x))
#
#
# class Downsample(nn.Module):
#     def __init__(self, dim):
#         super(Downsample, self).__init__()
#         self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# # building block modules
#
#
# class Block(nn.Module):
#     def __init__(self, dim, dim_out, groups=32, dropout=0):
#         super(Block, self).__init__()
#         # self.block = nn.Sequential(
#         self.bn = nn.BatchNorm2d(dim)
#         # self.act = nn.ReLU(inplace=True)
#         self.act = Swish()
#         # Swish(),
#         self.dropout = nn.Dropout(dropout) if dropout != 0 else nn.Identity()
#         self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)
#         # )
#
#     def forward(self, x):
#         x = self.bn(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         x = self.conv(x)
#         return x
#
#
# class ResnetBlock(nn.Module):
#     def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
#         super(ResnetBlock, self).__init__()
#         self.noise_func = FeatureWiseAffine(
#             noise_level_emb_dim, dim_out, use_affine_level)
#
#         self.block1 = Block(dim, dim_out, groups=norm_groups)
#         self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
#         self.res_conv = nn.Conv2d(
#             dim, dim_out, 1) if dim != dim_out else nn.Identity()
#
#     def forward(self, x, time_emb):
#         b, c, h, w = x.shape
#         h = self.block1(x)
#         h = self.noise_func(h, time_emb)
#         h = self.block2(h)
#         return h + self.res_conv(x)
#
#
# class SelfAttention(nn.Module):
#     def __init__(self, in_channel, n_head=1, norm_groups=32):
#         super(SelfAttention, self).__init__()
#
#         self.n_head = n_head
#
#         # self.norm = nn.GroupNorm(norm_groups, in_channel)
#         self.norm = nn.BatchNorm2d(in_channel)
#         self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
#         self.out = nn.Conv2d(in_channel, in_channel, 1)
#
#     def forward(self, input):
#         batch, channel, height, width = input.shape
#         n_head = self.n_head
#         head_dim = channel // n_head
#
#         norm = self.norm(input)
#         qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
#         query, key, value = qkv.chunk(3, dim=2)  # bhdyx
#
#         attn = torch.einsum(
#             "bnchw, bncyx -> bnhwyx", query, key
#         ).contiguous() / math.sqrt(channel)
#         attn = attn.view(batch, n_head, height, width, -1)
#         attn = torch.softmax(attn, -1)
#         attn = attn.view(batch, n_head, height, width, height, width)
#
#         out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
#         out = self.out(out.view(batch, channel, height, width))
#
#         return out + input
#
#
# class ResnetBlocWithAttn(nn.Module):
#     def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
#         super(ResnetBlocWithAttn, self).__init__()
#         self.with_attn = with_attn
#         self.res_block = ResnetBlock(
#             dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
#         if with_attn:
#             self.attn = SelfAttention(dim_out, norm_groups=norm_groups)
#
#     def forward(self, x, time_emb):
#         x = self.res_block(x, time_emb)
#         if(self.with_attn):
#             x = self.attn(x)
#         return x
#
#
# class UNet(nn.Module):
#     def __init__(
#         self,
#         in_channel=768,
#         out_channel=768,
#         inner_channel=768,
#         norm_groups=32,
#         channel_mults=(1, 1.5, 2),
#         attn_res=(7, 8),
#         res_blocks=3,
#         dropout=0,
#         with_noise_level_emb=True,
#         image_size=16,
#         conditional=True
#     ):
#         super(UNet, self).__init__()
#         if conditional:
#             ori_channel = in_channel
#             in_channel = in_channel
#         if with_noise_level_emb:
#             noise_level_channel = inner_channel
#             self.noise_level_mlp = nn.Sequential(
#                 PositionalEncoding(inner_channel),
#                 nn.Linear(inner_channel, inner_channel * 4),
#                 nn.ReLU(inplace=True), #nn.ReLU(inplace=True), Swish()
#                 nn.Linear(inner_channel * 4, inner_channel)
#             )
#         else:
#             noise_level_channel = None
#             self.noise_level_mlp = None
#
#         num_mults = len(channel_mults)
#         pre_channel = inner_channel
#         feat_channels = [pre_channel]
#         now_res = image_size
#         self.first_conv_rgb1 = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(in_channel),
#             # nn.ReLU(inplace=True)
#             Swish()
#         )
#         self.first_conv_rgb2 = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(in_channel),
#             # nn.ReLU(inplace=True)
#             Swish()
#         )
#         self.first_conv_tir1 = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(in_channel),
#             # nn.ReLU(inplace=True)
#             Swish()
#         )
#         self.first_conv_tir2 = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(in_channel),
#             # nn.ReLU(inplace=True)
#             Swish()
#         )
#         self.first_conv = nn.Sequential(
#             nn.Conv2d(in_channel * 3, inner_channel * 2, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(inner_channel * 2),
#             # nn.ReLU(inplace=True)
#             Swish()
#         )
#         downs = [nn.Conv2d(inner_channel * 2, inner_channel,
#                            kernel_size=3, padding=1)]
#         for ind in range(num_mults):
#             is_last = (ind == num_mults - 1)
#             use_attn = (now_res in attn_res)
#             channel_mult = int(inner_channel * channel_mults[ind])
#             for _ in range(0, res_blocks):
#                 downs.append(ResnetBlocWithAttn(
#                     pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
#                 feat_channels.append(channel_mult)
#                 pre_channel = channel_mult
#             if not is_last:
#                 downs.append(Downsample(pre_channel))
#                 feat_channels.append(pre_channel)
#                 now_res = now_res//2
#         self.downs = nn.ModuleList(downs)
#
#         self.mid = nn.ModuleList([
#             ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
#                                dropout=dropout, with_attn=True),
#             ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
#                                dropout=dropout, with_attn=False)
#         ])
#
#         ups = []
#         self.res = []
#         for ind in reversed(range(num_mults)):
#             is_last = (ind < 1)
#             use_attn = (now_res in attn_res)
#             channel_mult = int(inner_channel * channel_mults[ind])
#             for _ in range(0, res_blocks+1):
#                 ups.append(ResnetBlocWithAttn(
#                     pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
#                         dropout=dropout, with_attn=use_attn))
#                 pre_channel = channel_mult
#             if not is_last:
#                 now_res = now_res * 2
#                 ups.append(Upsample(pre_channel, now_res))
#                 self.res.append(now_res)
#
#         self.ups = nn.ModuleList(ups)
#
#         self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
#         self.norm_final = nn.BatchNorm2d(out_channel)
#
#
#     def forward(self, x, time):
#         t = self.noise_level_mlp(time) if exists(
#             self.noise_level_mlp) else None
#
#         # x = self.first_conv(x)
#         length = int(x.shape[1] / 3)
#         x_f = x[:,:length,:,:]
#         x_rgb = x[:,length:int(length*2),:,:]
#         x_tir = x[:,-length:,:,:]
#         x_rgb = self.first_conv_rgb2(self.first_conv_rgb1(x_rgb))
#         x_tir = self.first_conv_tir2(self.first_conv_tir1(x_tir))
#         x = torch.cat((torch.cat((x_f, x_rgb), dim=1), x_tir), dim=1)
#         x = self.first_conv(x)
#         feats = []
#         for layer in self.downs:
#             if isinstance(layer, ResnetBlocWithAttn):
#                 x = layer(x, t)
#             else:
#                 x = layer(x)
#             feats.append(x)
#         for layer in self.mid:
#             if isinstance(layer, ResnetBlocWithAttn):
#                 x = layer(x, t)
#             else:
#                 x = layer(x)
#         for layer in self.ups:
#             if isinstance(layer, ResnetBlocWithAttn):
#                 x = layer(torch.cat((x, feats.pop()), dim=1), t)
#             else:
#                 x = layer(x)
#
#         return self.norm_final(self.final_conv(x))
import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import matplotlib.pyplot as plt
from lib.model.models.WFCG import GAT
from lib.model.backbones.featurefusion_network import FeatureFusionNetwork
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

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super(PositionalEncoding, self).__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, int(out_channels*(1+self.use_affine_level)))
        )

    def forward(self, batch, x, noise_embed):
        # batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim, new_res):
        super(Upsample, self).__init__()
        # self.up = nn.Upsample(size=new_res, scale_factor=2, mode="nearest")
        self.up = nn.Upsample(size=new_res, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            # nn.BatchNorm2d(dim),
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super(ResnetBlock, self).__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(b, h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super(SelfAttention, self).__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        # self.norm = nn.BatchNorm2d(in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=12, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super(ResnetBlocWithAttn, self).__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=256,
        out_channel=256,
        inner_channel=256,
        norm_groups=32,
        channel_mults=(1, 1.5, 2),
        attn_res=(7, 8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=16,
        conditional=True
    ):
        super(UNet, self).__init__()
        if conditional:
            ori_channel = in_channel
            in_channel = in_channel * 3
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 3),
                Swish(),
                nn.Linear(inner_channel * 3, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = int(inner_channel * channel_mults[ind])
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        self.res = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = int(inner_channel * channel_mults[ind])
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                now_res = now_res * 2
                ups.append(Upsample(pre_channel, now_res))
                self.res.append(now_res)

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
        self.final_conv2 = Block(out_channel, default(out_channel, out_channel), groups=norm_groups)
        self.norm = nn.GroupNorm(norm_groups, out_channel)
        self.conv_GAT = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=4, padding=0)
        self.conv_11 = nn.Conv2d(in_channels=640, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.GroupNorm(norm_groups, 512)
        #self.selayer = SqueezeAndExcitation(channel=640,reduction=16)
        # self.norm = nn.BatchNorm2d(out_channel)

    def forward(self, x, time, fea_rgb, GAT_level1):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None
        # rgb = x[:,256:256+256,:,:]
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t) #[16, 512, 4, 4]
            else:
                x = layer(x)

        #fuse the first level feature of stage1
        GAT_level1 = self.conv_GAT(GAT_level1)
        concat = torch.cat((x, GAT_level1), dim=1)
        # concat = self.selayer(concat)
        x_add = self.conv_11(concat)
        x = self.norm1(x + x_add)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        # return self.norm(self.final_conv(x))
        # return self.norm(self.final_conv(x)+fea_rgb)
        return self.norm(self.final_conv2(self.final_conv(x)+fea_rgb))

    def feature2token(self, x):
        B, C, H, W = x.shape
        L = W * H
        tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
        return tokens
    def token2feature(self, tokens):
        B, L, D = tokens.shape
        H = W = int(L ** 0.5)
        x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
        return x
