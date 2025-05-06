import pprint
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn

from lib.model.backbones.MAE import mae_vit_base_patch16, mae_vit_small_patch16
from lib.model.backbones.featurefusion_network import FeatureFusionNetwork
from lib.model.backbones.t_SNE_visualize import draw_tsne

class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    N, C, _, _ =img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

def random_masking(x, x1, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore).unsqueeze(2)
    x_return = x * (1 - mask) + x1 * mask
    # ids_shuffle = ids_shuffle.detach().cpu().numpy()
    # ids_restore = ids_restore.detach().cpu().numpy()
    # x = x.detach().cpu().numpy()
    # mask = mask.detach().cpu().numpy()
    # x_masked = x_masked.detach().cpu().numpy()
    # x_masked2 = x_masked2.detach().cpu().numpy()

    return x_return, mask, ids_restore

class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1_1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        # self.conv1_2 = nn.Conv2d(in_channels=inplanes*2, out_channels=1, kernel_size=1, stride=1, padding=0)
        # self.conv2_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        # self.selfatten = FeatureFusionNetwork(d_model=768,
        #             dropout=0.1,
        #             nhead=1,
        #             dim_feedforward=2048,
        #             num_featurefusion_layers=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C/2), :, :].contiguous()
        # x0_ = x.contiguous()
        x0 = self.conv0_0(x0)

        x1 = x[:, int(C/2):, :, :].contiguous()
        # x1 = x.contiguous()

        # gate = self.sigmoid(self.conv1_2(x))
        # gate = self.avg_pool(gate)
        # binary = (gate > 0.5).float() #> threshold 1
        #
        # x_modality, _, _ = random_masking(feature2token(x0_), feature2token(x1), mask_ratio=0.3) #16,256,768
        # x_modality = token2feature(x_modality)
        # x_modality = self.selfatten(x_modality, x_modality)
        # x_modality = self.conv2_0(x_modality)

        x1 = self.conv0_1(x1)
        # x1 = self.relu(x1)

        x0 = self.fovea(x0) + x1# + x_modality
        # x0 = x_modality

        return self.conv1_1(x0)


class MAEEncode(nn.Module):

    def __init__(self,
                 arch: str,
                 train_flag: bool = False,
                 train_all: bool = False,
                 weights: str = None,
                 embed_dim=768,
                 train_layers: list = []
                 ):
        super(MAEEncode, self).__init__()

        if 'base' in arch:
            model = mae_vit_base_patch16()

            if weights is not None:
                print('load pretrain encoder from:', weights.split('/')[-1])
                ckp_dict = torch.load(weights, map_location='cpu')['model']
                ckp_dict = {k.replace('backbone.model.', ''): v for k, v in ckp_dict.items()}
                # ckp_dict = torch.load(weights, map_location='cpu')['net']
                # ckp_dict = {k.replace('backbone.', ''): v for k, v in ckp_dict.items()}
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in ckp_dict.items() if
                                   k in model_dict and v.shape == model_dict[k].shape}
                unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
                lost_param = [k for k, v in model_dict.items() if k not in ckp_dict or v.shape != ckp_dict[k].shape]
                print('unused param:')
                pprint.pprint(sorted(unused_param))
                print('lost_param:')
                pprint.pprint(sorted(lost_param))

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            else:
                ckp_dict = torch.load('/home/space/Documents/Experiments/BaseT/pretrain/mae_pretrain_vit_base_full.pth',
                                      map_location='cpu')['model']

                model_dict = model.state_dict()

                pretrained_dict = {k: v for k, v in ckp_dict.items() if
                                   k in model_dict and v.shape == model_dict[k].shape}
                unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
                lost_param = [k for k, v in model_dict.items() if k not in ckp_dict or v.shape != ckp_dict[k].shape]
                print('unused param:')
                pprint.pprint(sorted(unused_param))
                print('lost_param:')
                pprint.pprint(sorted(lost_param))

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

            self.patch_embed = deepcopy(model.patch_embed)
            self.pos_embed = deepcopy(model.pos_embed)  # fixed sin-cos embedding

            self.z_patch_embed = deepcopy(model.z_patch_embed)
            self.z_pos_embed = deepcopy(model.z_pos_embed)  # fixed sin-cos embedding

            # self.conv_dim = nn.Conv2d(in_channels=768*2, out_channels=768, kernel_size=1, stride=1,
            #                          padding=0)

            # self.patch_embed_prompt = deepcopy(model.patch_embed)
            #
            # self.z_patch_embed_prompt = deepcopy(model.z_patch_embed)

            self.cls_token = deepcopy(model.cls_token)
            self.blocks = deepcopy(model.blocks)
            self.norm = deepcopy(model.norm)

            self.patch_embed_p = deepcopy(model.patch_embed)
            self.pos_embed_p = deepcopy(model.pos_embed)  # fixed sin-cos embedding

            self.z_patch_embed_p = deepcopy(model.z_patch_embed)
            self.z_pos_embed_p = deepcopy(model.z_pos_embed)  # fixed sin-cos embedding

            self.cls_token_p = deepcopy(model.cls_token)
            self.norm_p = deepcopy(model.norm)

            self.pos_embed_x = deepcopy(model.pos_embed)  # fixed sin-cos embedding
            self.z_pos_embed_x = deepcopy(model.z_pos_embed)  # fixed sin-cos embedding

            self.cls_token_x = deepcopy(model.cls_token)
            self.norm_x = deepcopy(model.norm)

            self.out_stride = self.patch_embed.patch_size[0]
            self.out_channels_list = [self.cls_token.shape[-1]]

            # norm_layer = partial(nn.LayerNorm, eps=1e-6)
            # prompt_blocks = []
            # block_nums = 1
            # for i in range(block_nums):
            #     prompt_blocks.append(Prompt_block(inplanes=768, hide_channel=8, smooth=True)) #384
            # self.prompt_blocks = nn.Sequential(*prompt_blocks)
            # prompt_norms = []
            # for i in range(block_nums):
            #     prompt_norms.append(norm_layer(embed_dim))
            # self.prompt_norms = nn.Sequential(*prompt_norms)

            del model
            del ckp_dict

        elif 'small' in arch:
            model = mae_vit_small_patch16()

            if weights is not None:
                print('load pretrain encoder from:', weights.split('/')[-1])
                ckp_dict = torch.load(weights, map_location='cpu')['model']
                ckp_dict = {k.replace('backbone.model.', ''): v for k, v in ckp_dict.items()}
                model_dict = model.state_dict()

                pretrained_dict = {k: v for k, v in ckp_dict.items() if
                                   k in model_dict and v.shape == model_dict[k].shape}
                unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
                lost_param = [k for k, v in model_dict.items() if k not in ckp_dict or v.shape != ckp_dict[k].shape]
                print('unused param:')
                pprint.pprint(sorted(unused_param))
                print('lost_param:')
                pprint.pprint(sorted(lost_param))

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            else:
                raise NotImplementedError  # mae does not have small vit

            self.patch_embed = deepcopy(model.patch_embed)
            self.pos_embed = deepcopy(model.pos_embed)  # fixed sin-cos embedding

            self.z_patch_embed = deepcopy(model.z_patch_embed)
            self.z_pos_embed = deepcopy(model.z_pos_embed)  # fixed sin-cos embedding

            self.cls_token = deepcopy(model.cls_token)
            self.blocks = deepcopy(model.blocks)
            self.norm = deepcopy(model.norm)

            self.patch_embed_p = deepcopy(model.patch_embed)
            self.pos_embed_p = deepcopy(model.pos_embed)  # fixed sin-cos embedding

            self.z_patch_embed_p = deepcopy(model.z_patch_embed)
            self.z_pos_embed_p = deepcopy(model.z_pos_embed)  # fixed sin-cos embedding

            self.cls_token_p = deepcopy(model.cls_token)
            self.norm_p = deepcopy(model.norm)

            self.pos_embed_x = deepcopy(model.pos_embed)  # fixed sin-cos embedding
            self.z_pos_embed_x = deepcopy(model.z_pos_embed)  # fixed sin-cos embedding

            self.cls_token_x = deepcopy(model.cls_token)
            self.norm_x = deepcopy(model.norm)

            # norm_layer = partial(nn.LayerNorm, eps=1e-6)
            # prompt_blocks = []
            # block_nums = 1
            # for i in range(block_nums):
            #     prompt_blocks.append(Prompt_block(inplanes=384, hide_channel=8, smooth=True))
            # self.prompt_blocks = nn.Sequential(*prompt_blocks)
            # prompt_norms = []
            # for i in range(block_nums):
            #     prompt_norms.append(norm_layer(embed_dim))
            # self.prompt_norms = nn.Sequential(*prompt_norms)

            self.out_stride = self.patch_embed.patch_size[0]
            self.out_channels_list = [self.cls_token.shape[-1]]

            del model
            del ckp_dict
        else:
            raise NotImplementedError

        if train_flag:
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(False)

            if train_all:
                for name, parameter in self.named_parameters():
                    if 'pos_embed' not in name:  # fixed sin-cos embedding
                        parameter.requires_grad_(True)
            else:
                for id in train_layers:
                    for name, parameter in self.blocks[id].named_parameters():
                        parameter.requires_grad_(True)
                for name, parameter in self.norm.named_parameters():
                    parameter.requires_grad_(True)

        else:
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(False)

    def forward(self, z_color, z_ir, x1_color, x1_ir):
        # embed patches
        z_color = self.z_patch_embed(z_color)
        z_ir = self.z_patch_embed_p(z_ir)

        # z_color = self.patch_embed(z_color)
        # z_ir = self.patch_embed(z_ir)
        x1_color = self.patch_embed(x1_color)
        x1_ir = self.patch_embed_p(x1_ir)  # [32, 196, 768]

        # draw_tsne(x1_color, x1_ir)

        # z = z_color + z_ir
        # x1 = x1_color + x1_ir

        # add pos embed w/o cls token
        x1_color = x1_color + self.pos_embed[:, 1:, :]
        # z = z + self.pos_embed[:, 1:, :]
        z_color = z_color + self.z_pos_embed[:, 0:, :]

        # add pos embed w/o cls token
        x1_ir = x1_ir + self.pos_embed_p[:, 1:, :]
        z_ir = z_ir + self.z_pos_embed_p[:, 0:, :]

        # # add pos embed w/o cls token
        # x1_x = x1_x + self.pos_embed_x[:, 1:, :]
        # z_x = z_x + self.z_pos_embed_x[:, 0:, :]

        len_z = z_color.shape[1]
        len_x = x1_color.shape[1]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens_x_color = cls_token.expand(x1_color.shape[0], -1, -1)
        x_color = torch.cat((cls_tokens_x_color, z_color, x1_color), dim=1)
        for blk in self.blocks:
            x_color = blk(x_color)

        x_color = self.norm(x_color)
        z_color = x_color[:, 1:1 + len_z]
        x1_color = x_color[:, 1 + len_z:1 + len_z + len_x]

        # append cls token
        cls_token_p = self.cls_token_p + self.pos_embed_p[:, :1, :]
        cls_tokens_x_ir = cls_token_p.expand(x1_ir.shape[0], -1, -1)
        x_ir = torch.cat((cls_tokens_x_ir, z_ir, x1_ir), dim=1)
        for blk in self.blocks:
            x_ir = blk(x_ir)

        x_ir = self.norm_p(x_ir)
        z_ir = x_ir[:, 1:1 + len_z]
        x1_ir = x_ir[:, 1 + len_z:1 + len_z + len_x]

        # x_color_feat = token2feature(self.prompt_norms[0](x1_color))  # [32, 768, 14, 14]
        # z_color_feat = token2feature(self.prompt_norms[0](z_color))  # [32, 768, 7, 7]
        # x_ir_feat = token2feature(self.prompt_norms[0](x1_ir))  # [32, 768, 14, 14]
        # z_ir_feat = token2feature(self.prompt_norms[0](z_ir))  # [32, 768, 7, 7]
        # x_feat = torch.cat([x_color_feat, x_ir_feat], dim=1)
        # z_feat = torch.cat([z_color_feat, z_ir_feat], dim=1)
        # x_feat = self.prompt_blocks[0](x_feat)
        # z_feat = self.prompt_blocks[0](z_feat)
        # x1_p = feature2token(x_feat)
        # z_p = feature2token(z_feat)

        # z = z_color + z_ir
        # x1 = x1_color + x1_ir

        return z_color, x1_color, z_ir, x1_ir
        # else:
        #     x2 = x[:, 1 + len_z + len_x:1 + len_z + 2*len_x]
        #     # x3 = x[:, 1 + len_z + 2*len_x:]
        #     return z, x1, x2




def build_backbone(_args):
    model = MAEEncode(arch=_args.arch, train_flag=_args.lr_mult > 0, train_all=_args.train_all,
                      weights=_args.weights, train_layers=_args.train_layers)

    return model

def token2feature(tokens):
    B,L,D=tokens.shape
    H=W=int(L**0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    return x

def feature2token(x):
    B,C,H,W = x.shape
    L = W*H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens


if __name__ == '__main__':
    from config.cfg_translation_track import cfg as exp

    backbone = build_backbone(exp.model.backbone)

    x = torch.rand(1, 3, 224, 224)
    z = torch.rand(1, 3, 112, 112)
    ys = backbone(x, z)
    print([_y.shape for _y in ys])

    from ptflops import get_model_complexity_info


    def prepare_input(resolution):
        input_dict = {
            'x': x,
            'z': z,
        }

        return input_dict


    flops, params = get_model_complexity_info(backbone,
                                              input_res=(None,),
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
    #       - Flops:  21.04 GMac
    #       - Params: 42.53 M
