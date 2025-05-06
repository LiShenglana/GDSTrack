# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TransT FeatureFusionNetwork class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch.nn.functional as F
import torch
from torch import nn, Tensor
import math


class FeatureFusionNetwork(nn.Module):

    def __init__(self, d_model=512, dim_feedforward=256, dropout=0.1, activation="relu"):
        super().__init__()
        self.dim_feedforward = dim_feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, dim_feedforward)

        # self.norm1 = nn.LayerNorm(dim_feedforward)
        # self.norm2 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, k=256, simi='QKV'):
        if simi == 'QKV':
            Q = self.linear1(src_temp)
            K = self.linear2(src_temp)
            attentions = torch.bmm(Q, K.transpose(1, 2))
            attentions = attentions / math.sqrt(self.dim_feedforward)
            # mean = attentions.mean(dim=(1,2), keepdim=True)
            # std = attentions.std(dim=(1,2), keepdim=True)
            # attentions = (attentions - mean) / (std + 1e-8)
            attentions = F.softmax(attentions, dim=-1)

            _, topk_indices = torch.topk(attentions, k, dim=2)  # [batch_size, num_tokens, K]
            adjacency = torch.zeros_like(attentions)
            adjacency.scatter_(2, topk_indices, 1)

            adjacency = (adjacency + adjacency.transpose(1 ,2)) / 2
        elif simi == 'cosine':
            # 特征归一化
            normalized_features = F.normalize(src_temp, p=2, dim=2)  # [batch_size, num_tokens, feature_dim]

            # 计算余弦相似度
            similarity = torch.bmm(normalized_features,
                                   normalized_features.transpose(1, 2))  # [batch_size, num_tokens, num_tokens]

            # 对每个token选择K个最相似的邻居
            _, topk_indices = torch.topk(similarity, k, dim=2)  # [batch_size, num_tokens, K]

            # 初始化邻接矩阵为0
            adjacency = torch.zeros_like(similarity)

            # 使用scatter_方法设置邻接关系
            adjacency.scatter_(2, topk_indices, 1)
            adjacency = (adjacency + adjacency.transpose(1, 2)) / 2
        elif simi == 'cosine+QKV':
            #QKV
            Q = self.linear1(src_temp)
            K = self.linear2(src_temp)
            attentions = torch.bmm(Q, K.transpose(1, 2))
            attentions = attentions / math.sqrt(self.dim_feedforward)


            #cosine
            # 特征归一化
            normalized_features = F.normalize(src_temp, p=2, dim=2)  # [batch_size, num_tokens, feature_dim]

            # 计算余弦相似度
            similarity = torch.bmm(normalized_features,
                                   normalized_features.transpose(1, 2))  # [batch_size, num_tokens, num_tokens]

            #mask
            similarity = similarity + attentions
            threshold = 0.5
            similarity = (similarity + 1) / 2  #[-1,1]->[0,1]
            mask = (similarity > threshold).float()
            attentions = attentions.masked_fill(mask == 0, float('-inf'))
            attentions = F.softmax(attentions, dim=-1) #[0,1]
            attentions = attentions + similarity

            #KNN
            # 对每个token选择K个最相似的邻居
            _, topk_indices = torch.topk(attentions, k, dim=2)  # [batch_size, num_tokens, K]

            # 初始化邻接矩阵为0
            adjacency = torch.zeros_like(attentions)

            # 使用scatter_方法设置邻接关系
            adjacency.scatter_(2, topk_indices, 1)
            adjacency = (adjacency + adjacency.transpose(1, 2)) / 2
        return adjacency

class wrap(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)
        #
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.activation = _get_activation_fn(activation)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, src_search, mask_temp=None, mask_search=None, pos_temp=None, pos_search=None):

        src_temp = src_temp.permute(1, 0, 2)
        src_search = src_search.permute(1, 0, 2)

        hs = self.encoder(src1=src_temp, src2=src_search,
                                                  src1_key_padding_mask=mask_temp,
                                                  src2_key_padding_mask=mask_search,
                                                  pos_src1=pos_temp,
                                                  pos_src2=pos_search) #(64, 32, 256)

        hs = hs.permute(1, 0, 2) #(32, 64, 256)
        opt = (hs.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous() #(32, 1, 256, 64)
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, int(HW**0.5), int(HW**0.5)) #(32, 256, 8, 8)
        return opt_feat


class Encoder(nn.Module):

    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):
        # output1 = src1
        # output2 = src2

        for layer in self.layers:
            output1 = layer(src1, src2, src1_mask=src1_mask,
                                     src2_mask=src2_mask,
                                     src1_key_padding_mask=src1_key_padding_mask,
                                     src2_key_padding_mask=src2_key_padding_mask,
                                     pos_src1=pos_src1, pos_src2=pos_src2)

        return output1

class FeatureFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2,
                     src1_mask: Optional[Tensor] = None,
                     src2_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     src2_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None,
                     pos_src2: Optional[Tensor] = None):
        # q1 = k1 = self.with_pos_embed(src1, pos_src1)
        # src12 = self.self_attn1(q1, k1, value=src1, attn_mask=src1_mask,
        #                        key_padding_mask=src1_key_padding_mask)[0]
        # src1 = src1 + self.dropout11(src12)
        # src1 = self.norm11(src1)
        #
        # q2 = k2 = self.with_pos_embed(src2, pos_src2)
        # src22 = self.self_attn2(q2, k2, value=src2, attn_mask=src2_mask,
        #                        key_padding_mask=src2_key_padding_mask)[0]
        # src2 = src2 + self.dropout21(src22)
        # src2 = self.norm21(src2)


        src12 = self.multihead_attn1(query=self.with_pos_embed(src1, pos_src1),
                                   key=self.with_pos_embed(src2, pos_src2),
                                   value=src2)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = self.norm13(src1)

        return src1

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):

        return self.forward_post(src1, src2, src1_mask, src2_mask,
                                 src1_key_padding_mask, src2_key_padding_mask, pos_src1, pos_src2)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_featurefusion_network(settings):
    return FeatureFusionNetwork(
        d_model=settings.hidden_dim,
        dropout=settings.dropout,
        nhead=settings.nheads,
        dim_feedforward=settings.dim_feedforward,
        num_featurefusion_layers=settings.featurefusion_layers
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
