import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.model.layers.frozen_bn import FrozenBatchNorm2d
from copy import deepcopy
import numpy as np

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

class Center(nn.Module, ):
    def __init__(self, args, freeze_bn=False):
        super(Center, self).__init__()
        self.cfg = args
        self.inplanes = self.cfg.in_channels
        self.channel = self.cfg.inter_channels
        self.feat_h, self.feat_w = self.cfg.search_size

        self.stride = self.cfg.stride
        assert self.feat_h == self.feat_w, "not support non-square feature map"
        self.feat_sz = self.feat_h
        self.img_sz = self.feat_sz * self.stride
        # self.feat_sz = feat_sz
        # self.stride = stride
        # self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = conv(self.inplanes, self.channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(self.channel, self.channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(self.channel // 2, self.channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(self.channel // 4, self.channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(self.channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = conv(self.inplanes, self.channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(self.channel, self.channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(self.channel // 2, self.channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(self.channel // 4, self.channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(self.channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(self.inplanes, self.channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(self.channel, self.channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(self.channel // 2, self.channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(self.channel // 4, self.channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(self.channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)
        bbox = self.box_cxcywh_to_xyxy(bbox).view(-1, 4) # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        if return_score:
            return bbox, max_score
        return bbox

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset