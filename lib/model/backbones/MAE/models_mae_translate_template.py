# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from lib.model.backbones.MAE.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3, #img_size=224
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # -------------------- new --------------------
        self.z_patch_embed = PatchEmbed(img_size // 2, patch_size, in_chans, embed_dim)
        z_num_patches = self.z_patch_embed.num_patches
        self.z_pos_embed = nn.Parameter(torch.zeros(1, z_num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # ---------------------------------------------

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # -------------------- new --------------------
        self.z_decoder_pos_embed = nn.Parameter(torch.zeros(1, z_num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # ---------------------------------------------

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # -------------------- new --------------------
        pos_embed = get_2d_sincos_pos_embed(self.z_pos_embed.shape[-1], int(self.z_patch_embed.num_patches ** .5), cls_token=False)
        self.z_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.z_patch_embed.num_patches ** .5), cls_token=True)
        self.z_decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # ---------------------------------------------

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        w = self.z_patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
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
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, z, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        z = self.z_patch_embed(z)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        z = z + self.z_pos_embed[:, 0:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio=mask_ratio)
        z, z_mask, z_ids_restore = self.random_masking(z, mask_ratio)

        len_z = z.shape[1]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, z, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        z = x[:, :1 + len_z]
        x = torch.cat((x[:, :1], x[:, 1 + len_z:]), dim=1)  # add cls token for using the same decoder function
        return z, z_mask, z_ids_restore, x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        if x.shape[1] == self.patch_embed.num_patches + 1:
            x = x + self.decoder_pos_embed
        else:
            x = x + self.z_decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, temps, targets, mask_ratio):
        latent_z, mask_z, ids_restore_z, latent_x, mask_x, ids_restore_x = self.forward_encoder(imgs, temps, mask_ratio)

        if targets.shape == temps.shape:
            pred_z = self.forward_decoder(latent_z, ids_restore_z)  # [N, L, p*p*3]
            loss_z = self.forward_loss(targets, pred_z, mask_z)

            pred_x = self.forward_decoder(latent_x, ids_restore_x)  # [N, L, p*p*3]
            loss_x = self.forward_loss(imgs, pred_x, mask_x)

            return loss_z, pred_z, mask_z, loss_x, pred_x
        elif targets.shape == imgs.shape:
            pred_x = self.forward_decoder(latent_x, ids_restore_x)  # [N, L, p*p*3]
            loss_x = self.forward_loss(imgs, pred_x, mask_x)

            return loss_x, pred_x

    # ========================== masking ratio is 0 =======================
    # def forward_encoder(self, x, z, mask_ratio):
    #     # embed patches
    #     x = self.patch_embed(x)
    #     z = self.z_patch_embed(z)
    #
    #     # add pos embed w/o cls token
    #     x = x + self.pos_embed[:, 1:, :]
    #     z = z + self.z_pos_embed[:, 0:, :]
    #
    #     len_z = z.shape[1]
    #
    #     # append cls token
    #     cls_token = self.cls_token + self.pos_embed[:, :1, :]
    #     cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    #     x = torch.cat((cls_tokens, z, x), dim=1)
    #
    #     # apply Transformer blocks
    #     for blk in self.blocks:
    #         x = blk(x)
    #     x = self.norm(x)
    #
    #     z = x[:, :1 + len_z]
    #     x = torch.cat((x[:, :1], x[:, 1 + len_z:]), dim=1)
    #
    #     return z, x
    #
    # def forward_decoder(self, x):
    #     # embed tokens
    #     x = self.decoder_embed(x)
    #
    #     # add pos embed
    #     if x.shape[1] == self.patch_embed.num_patches + 1:
    #         x = x + self.decoder_pos_embed
    #     else:
    #         x = x + self.z_decoder_pos_embed
    #
    #     # apply Transformer blocks
    #     for blk in self.decoder_blocks:
    #         x = blk(x)
    #     x = self.decoder_norm(x)
    #
    #     # predictor projection
    #     x = self.decoder_pred(x)
    #
    #     # remove cls token
    #     x = x[:, 1:, :]
    #
    #     return x
    #
    # def forward_loss(self, imgs, pred):
    #     """
    #     imgs: [N, 3, H, W]
    #     pred: [N, L, p*p*3]
    #     mask: [N, L], 0 is keep, 1 is remove,
    #     """
    #     target = self.patchify(imgs)
    #     if self.norm_pix_loss:
    #         mean = target.mean(dim=-1, keepdim=True)
    #         var = target.var(dim=-1, keepdim=True)
    #         target = (target - mean) / (var + 1.e-6) ** .5
    #
    #     loss = (pred - target) ** 2
    #     loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    #
    #     return loss.mean()
    #
    # def forward(self, imgs, temps, targets, mask_ratio):
    #     latent_z, latent_x = self.forward_encoder(imgs, temps, mask_ratio)
    #
    #     pred_z = self.forward_decoder(latent_z)  # [N, L, p*p*3]
    #     loss_z = self.forward_loss(targets, pred_z)
    #
    #     pred_x = self.forward_decoder(latent_x)  # [N, L, p*p*3]
    #     loss_x = self.forward_loss(imgs, pred_x)
    #
    #     return loss_z, pred_z, None, loss_x, pred_x


def mae_vit_small_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_small_patch16 = mae_vit_small_patch16

if __name__ == '__main__':
    model = mae_vit_base_patch16()
    x = torch.rand(1, 3, 224, 224)
    z = torch.rand(1, 3, 112, 112)
    loss, y, mask, loss_x, pred_x = model(x, z, z, mask_ratio=0.25)

    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    z = torch.einsum('nchw->nhwc', z)
    # masked image
    im_masked = z * (1 - mask)
    # MAE reconstruction pasted with visible patches
    im_paste = z * (1 - mask) + y * mask
