import importlib
from typing import Union, Dict, Any

import time

import cv2
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torchvision.ops import box_convert
import torch.nn.functional as F
#from lib.model.prroi_pool.functional import prroi_pool2d
from lib.model.models._model import Model as BaseModel
from lib.vis.visdom_cus import Visdom
import os
from lib.model.models.lddifuse.lddifuse import LDDiffuse
from lib.model.models.WFCG import GAT
from torchprofile import profile_macs
from lib.model.backbones.featurefusion_network import FeatureFusionNetwork

class LearnableAdjacency(nn.Module):
    def __init__(self, num_tokens):
        super(LearnableAdjacency, self).__init__()
        self.adj_param = nn.Parameter(torch.full((num_tokens, num_tokens), -3.0))
        with torch.no_grad():
            self.adj_param.fill_diagonal_(3.0)
    def forward(self):
        adjacency_matrix = torch.sigmoid(self.adj_param)
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.t()) / 2
        # adjacency_matrix = adjacency_matrix.masked_fill(
        #     torch.eye(self.adj_param.size(0), device=self.adj_param.device).bool(), 0)
        return adjacency_matrix
    def regularization_loss(self):
        return torch.sum(torch.abs(torch.sigmoid(self.adj_param)))

class Model(BaseModel):

    def __init__(self, args):
        super(Model, self).__init__()

        self.debug = False
        self.use_visdom = False
        self.pretrained_param = None
        self.last_fea = None
        self.cfg = args.model
        self.cfg_all = args
        self.fusemode = 'GAT' #ADD or GAT
        # build backbone
        backbone_module = importlib.import_module('lib.model.backbones')
        self.backbone = getattr(backbone_module, self.cfg.backbone.type)(self.cfg.backbone)
        self.cfg.backbone.out_stride = self.backbone.out_stride
        self.stage = self.cfg_all.trainer.stage

        # build neck
        # neck_module = importlib.import_module('lib.model.necks')
        self.cfg.neck.search_size = [sz // self.cfg.backbone.out_stride for sz in self.cfg.backbone.search_size]
        self.cfg.neck.template_size = [sz // self.cfg.backbone.out_stride for sz in self.cfg.backbone.template_size]
        # self.cfg.neck.in_channels_list = [c for c in self.backbone.out_channels_list]
        # self.neck = getattr(neck_module, self.cfg.neck.type)(self.cfg.neck)

        # build head
        head_module = importlib.import_module('lib.model.heads')
        self.cfg.head.search_size = self.cfg.neck.search_size
        self.cfg.head.stride = self.cfg.backbone.out_stride
        self.head = getattr(head_module, self.cfg.head.type)(self.cfg.head)
        # self.box_head = getattr(head_module, self.cfg.head.type)(self.cfg.head)
        # self.fuse_search = conv(hidden_dim * 2, hidden_dim)
        if self.fusemode == 'GAT':
            self.generate_adjacency = FeatureFusionNetwork(
                d_model=768,
                dropout=0.1,
                dim_feedforward=2048
            )
            # self.adjacency = LearnableAdjacency(num_tokens=512)
            self.GAT_Branch = GAT(nfeat=768, nhid=128, nout=256, nheads=1, dropout=0.4, alpha=0.2, num_nodes=256)
            self.projector1 = nn.Linear(768, 256)
        if self.stage == 'stage3':
            self.GAT_Branch2 = GAT(nfeat=256, nhid=128, nout=256, nheads=1, dropout=0.4, alpha=0.2, num_nodes=256)
        # self.projector2 = nn.Linear(1792, 1536)
        if self.stage == 'stage2' or self.stage == 'stage3':
            self.projector2 = nn.Linear(768, 256)
            self.lddiffuse_x = LDDiffuse(Total=1000, Sample=1, image_size=16)
            self.head_diffuse = getattr(head_module, self.cfg.head.type)(self.cfg.head)
            # for p in self.parameters():
            #     if p.dim() > 1:
            #         nn.init.xavier_uniform_(p)

        # build criterion
        criteria_module = importlib.import_module('lib.criteria')
        self.criteria = getattr(criteria_module, self.cfg.criterion.type)(self.cfg.criterion)

        config = TrackerConfig()
        # self.target = torch.tensor(config.y).cuda().unsqueeze(0).unsqueeze(0).repeat(self.cfg_all.data.batch_size, 1, 1, 1)
        if self.debug:
            if not self.use_visdom:
                self.save_dir = 'debug'
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        self.next_seq = False
        if debug > 0 and visdom_info.get('use_visdom', True):
            try:
                self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                                     visdom_info=visdom_info, env='MAT')

            except:
                time.sleep(0.5)
                print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                      '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

            elif data['key'] == 'n':
                self.next_seq = True

    def _cls_loss(self, pred, label, select):
        if len(select.size()) == 0:
            return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)
    def _weighted_BCE(self, pred, label):
        pred = pred.view(-1)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze().cuda()
        neg = label.data.eq(0).nonzero().squeeze().cuda()

        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)

        return loss_pos * 0.5 + loss_neg * 0.5
    def calc_score_center(self, score_lt1, score_br1):
        B, L = score_lt1.shape
        H = W = int(L ** 0.5)
        # score_lt1 = score_lt1.cpu().detach().numpy()
        # score_br1 = score_br1.cpu().detach().numpy()
        # score_lt1 = score_lt1.view((-1, 1, H, W))
        # score_br1 = score_br1.view((-1, 1, H, W))
        peak1, index1 = torch.max(score_lt1, 1)
        score_lt1 = score_lt1.view(B, H, W)
        score_center = torch.zeros(score_lt1.shape)
        r_max1 = index1 / H
        c_max1 = index1 % W

        peak2, index2 = torch.max(score_br1, 1)
        r_max2 = index2 / H
        c_max2 = index2 % W

        r_shift = (r_max2 - r_max1) * 0.5
        c_shift = (c_max2 - c_max1) * 0.5
        for j in range(B):
            shift_label = torch.roll(score_lt1[j], int(r_shift[j]), 0)
            score_center[j, ...] = torch.roll(shift_label, int(c_shift[j]), 1)
        score_center = torch.Tensor(score_center).view(B, H, W).cuda()
        return score_center

    def Region_mask(self, search_feat, pred_boxes):
        # search_feat = self.input_proj(search_feat)
        # template_feat = self.input_proj(template_feat)
        # last_feat = self.input_proj(last_feat)
        pred_boxes = pred_boxes * 16
        x = torch.tensor(range(0, 15)).cuda()
        y = torch.tensor(range(0, 15)).cuda()
        x1, y1 = torch.meshgrid(x, y)
        grid = torch.stack((x1, y1), 2)
        grid = grid.view(-1, 2)
        B, C, H, W = search_feat.shape
        mask = torch.zeros(B, H, W).cuda()
        for x, y in grid:
            z1 = torch.min(x + 1, pred_boxes[:, 2]) - torch.max(x, pred_boxes[:, 0])
            z2 = torch.min(y + 1, pred_boxes[:, 3]) - torch.max(y, pred_boxes[:, 1])
            z1[z1 < 0] = 0
            z2[z2 < 0] = 0
            z = ((z1 * z2) / 1).cuda()
            mask[:, y:y + 1, x:x + 1] = z.view(-1, 1, 1)
        roi = torch.mul(search_feat, mask.unsqueeze(1))
        return roi

    def forward(self, input_dict: Dict[str, Union[Tensor, Any]]):

        device = next(self.parameters()).device

        templates_color: Tensor = input_dict['template_color'].to(device)
        templates_ir: Tensor = input_dict['template_ir'].to(device)
        searchs_color: Tensor = input_dict['search_color'].to(device)
        searchs_ir: Tensor = input_dict['search_ir'].to(device)
        s_bbox: Tensor = input_dict['template_ori'].to(device)
        searchs3_color: Tensor = input_dict['search3_color'].to(device)
        searchs3_ir: Tensor = input_dict['search3_ir'].to(device)
        s3_bbox: Tensor = input_dict['s3_box'].to(device)

        # ----------- backbone feature -------------------------
        # s0_feat, s1_feat, s2_feat, s3_feat = self.get_backbone_feature(templates_color, templates_ir, searchs1_color, searchs1_ir, searchs2_color, searchs2_ir, searchs3_color, searchs3_ir)
        # s0_feat1, s1_feat = self.get_backbone_feature(templates_color, templates_ir, searchs1_color, searchs1_ir)
        # s0_feat2, s2_feat = self.get_backbone_feature(templates_color, templates_ir, searchs2_color, searchs2_ir)
        z_color, x1_color, z_ir, x1_ir = self.get_backbone_feature(templates_color, templates_ir, searchs_color, searchs_ir)
        _, x3_color, _, x3_ir = self.get_backbone_feature(templates_color, templates_ir, searchs3_color,
                                                                   searchs3_ir)
        if self.fusemode == 'ADD':
            s_feat = x1_color + x1_ir
        elif self.fusemode == 'GAT':
            inputs = torch.cat([x1_color, x1_ir], dim=1) #[16,512,768]
            A = self.generate_adjacency(inputs, simi='cosine+QKV')
            _, outputs = self.GAT_Branch(inputs, A=A) #[16,512,256]
            B, WH, C = outputs.shape
            s_feat = outputs[:, 0:int(WH/2), :] + outputs[:, int(WH/2):, :] # [16,256,256]
            s_feat = s_feat + self.projector1(x1_color + x1_ir)
        s_feat = self.token2feature(s_feat)
        if self.stage == 'stage1':
            pred_boxes1, score_lt, score_br = self.head(s_feat)
        filt_bg = False
        if filt_bg:
            pred_boxes_noise, score_lt, score_br = self.head(s_feat)
            B, _, H, W = s_feat.shape
            x1 = (pred_boxes_noise[:, 0] * W).long()
            y1 = (pred_boxes_noise[:, 1] * H).long()
            x2 = (pred_boxes_noise[:, 2] * W).long()
            y2 = (pred_boxes_noise[:, 3] * H).long()
            mask = torch.zeros(B, H, W, dtype=torch.float32).cuda()
            for i in range(B):
                mask[i, y1[i]:y2[i], x1[i]:x2[i]] = 1
            mask = mask.unsqueeze(1)
            masked_s_feat = s_feat * mask
        # pred_boxes0, score_lt, score_br = self.head(s_feat)
        if self.stage == 'stage2':
            inputs3 = torch.cat([x3_color, x3_ir], dim=1)  # [16,512,768]
            A3 = self.generate_adjacency(inputs3, simi='cosine+QKV')
            x1, outputs3 = self.GAT_Branch(inputs3, A=A3)  # [16,512,256]
            B1, WH1, C1 = x1.shape
            GAT_level1 = x1[:, 0:int(WH1 / 2), :] + x1[:, int(WH1 / 2):, :]  # [16,256,128]
            GAT_level1 = self.token2feature(GAT_level1)  #[16, 128, 16, 16]
            B, WH, C = outputs3.shape
            s_feat3 = outputs3[:, 0:int(WH / 2), :] + outputs3[:, int(WH / 2):, :]  # [16,256,256]
            s_feat3 = s_feat3 + self.projector1(x3_color + x3_ir)
            s_feat3 = self.token2feature(s_feat3)
            pred_boxes0, score_lt, score_br = self.head(s_feat3)
            loss_lddifuse_x, out_x = self.lddiffuse_x(s_feat3, s_feat, GAT_level1, self.token2feature(self.projector2(x3_color)),
                                                self.token2feature(self.projector2(x3_ir)))
            loss_lddifuse_x = loss_lddifuse_x / (out_x.shape[0] * out_x.shape[1] * out_x.shape[2] * out_x.shape[3])
            pred_boxes2, score_lt, score_br = self.head_diffuse(out_x)
        # if self.stage == 'stage3':
        #     inputs2 = torch.cat([self.feature2token(s_feat), self.feature2token(out_x)], dim=1)  # [16,512,256]
        #     outputs2 = self.GAT_Branch2(inputs2, A='cosine')
        #     B, WH, C = outputs2.shape
        #     s_feat = outputs2[:, 0:int(WH / 2), :] + outputs2[:, int(WH / 2):, :]  # [16,256,256]
        #     s_feat = self.token2feature(s_feat)
        # output_back = self.neck(s3_feat, s0_feat3)
        # s3_feat = self.projector(s3_feat)


        test_outbox = False
        if test_outbox:
            t_box = s3_bbox[5] * 256
            s3_bbox = pred_boxes[5] * 256
            # s2_box = pred_boxes2[5] * 256
            im_show = cv2.cvtColor(searchs3_color[5].permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.rectangle(im_show, (int(t_box[0]), int(t_box[1])), (int(t_box[2]), int(t_box[3])), (0, 255, 0),
                          3)
            im_show2 = cv2.cvtColor(searchs3_color[5].permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.rectangle(im_show2, (int(s3_bbox[0]), int(s3_bbox[1])), (int(s3_bbox[2]), int(s3_bbox[3])), (0, 255, 0),
                          3)
            # im_show3 = cv2.cvtColor(searchs2_color[5].permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
            # cv2.rectangle(im_show3, (int(s2_box[0]), int(s2_box[1])), (int(s2_box[2]), int(s2_box[3])), (0, 255, 0),
            #               3)
            cv2.imshow('searchs_color', im_show)
            cv2.imshow('searchs3_color', im_show2)
            # cv2.imshow('search2_img_color', im_show3)
            cv2.waitKey()

        if self.debug:
            if self.use_visdom:
                for index in range(score_center0.shape[0]):

                    # self.visdom.register(mask1[index].squeeze().view(16, 16), 'heatmap', 1,
                    #                      'mask1-{}'.format(time.strftime('%M:%S')))
                    # self.visdom.register(mask2[index].squeeze().view(16, 16), 'heatmap', 1,
                    #                      'mask2-{}'.format(time.strftime('%M:%S')))
                    # self.visdom.register(mask3[index].squeeze().view(16, 16), 'heatmap', 1,
                    #                      'mask3-{}'.format(time.strftime('%M:%S')))
                    self.visdom.register(label_t[index].view(16, 16), 'heatmap', 1,
                                         'label_t-{}'.format(time.strftime('%M:%S')))
                    # self.visdom.register(label_s1[index].view(16, 16), 'heatmap', 1,
                    #                      'label_s1-{}'.format(time.strftime('%M:%S')))
                    # self.visdom.register(label_s2[index].view(16, 16), 'heatmap', 1,
                    #                      'label_s2-{}'.format(time.strftime('%M:%S')))
                    self.visdom.register(score_center0[index].view(16, 16), 'heatmap', 1,
                                         'score_center0-{}'.format(time.strftime('%M:%S')))
                    # self.visdom.register(score_lt3[index].view(16, 16), 'heatmap', 1,
                    #                      'score_lt3-{}'.format(time.strftime('%M:%S')))
                    # self.visdom.register(score_br3[index].view(16, 16), 'heatmap', 1,
                    #                      'score_br3-{}'.format(time.strftime('%M:%S')))

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        # ----------- compute loss -------------------------
        loss_dict = dict()
        metric_dict = dict()

        if self.stage == 'stage2':
            bbox_losses0, metrics0 = self.criteria([pred_boxes2, None], [s3_bbox, None])
            # bbox_losses, metrics = self.criteria([pred_boxes2, None], [pred_boxes0, None])
            total_loss = bbox_losses0[0] + 1 * (loss_lddifuse_x)# + bbox_losses[0]
        else:
            bbox_losses0, metrics0 = self.criteria([pred_boxes1, None], [s_bbox, None])
            total_loss = bbox_losses0[0]

        if torch.isnan(total_loss):
            print("debug")
        for d in bbox_losses0[1:]:
            loss_dict.update(d)

        # loss_dict1.update({'sim1', losses1.item()})
        #
        # loss_dict1.update({'sim2', losses2.item()})

        for d in metrics0:
            metric_dict.update(d)

        return total_loss, [loss_dict, metric_dict]

    def token2feature(self, tokens):
        B, L, D = tokens.shape
        H = W = int(L ** 0.5)
        x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
        return x

    def weight_score(self, score_center, label_t):
        threshold = 0.004
        gama = 0.3
        binary_map = (score_center > threshold).float()
        weight = (1 - torch.mean(binary_map, dim=(1, 2))/256) * gama
        cls_loss = self.criterion(score_center, label_t, weight)
        return cls_loss

    def prpool_feature(self, features, bboxs, spatial_scale = 1.0):
        batch_index = torch.arange(0, features.shape[0]).view(-1, 1).float().cuda()
        bboxs_index = torch.cat((batch_index, bboxs), dim=1)
        return prroi_pool2d(features, bboxs_index, 8, 8, spatial_scale) #1.0

    def token2feature(self, tokens):
        B, L, D = tokens.shape
        H = W = int(L ** 0.5)
        x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
        return x
    def feature2token(self, x):
        B, C, H, W = x.shape
        L = W * H
        tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
        return tokens

    def track(self, images_color: Tensor, images_ir: Tensor, templates_color: Tensor, templates_ir: Tensor, templates_color_s: Tensor, templates_ir_s: Tensor, first_box: Tensor, last_box: Tensor, **kwargs):

        ns, _, hs, ws = images_color.shape

        # ----------- backbone feature -------------------------
        # macs = profile_macs(self.get_backbone_feature, (templates_color.cpu(), templates_ir.cpu(), images_color.cpu(), images_ir.cpu()))
        # flops = (2 * macs) / 1e9
        # print(f"GFLOPS: {flops:,}")
        z_color, x1_color, z_ir, x1_ir = self.get_backbone_feature(templates_color, templates_ir, images_color, images_ir)
        # s0_feat = z_color + z_ir
        if self.fusemode == 'ADD':
            s3_feat = self.token2feature(self.projector(x1_color + x1_ir))
        elif self.fusemode == 'GAT':
            # A = self.adjacency()
            inputs = torch.cat([x1_color, x1_ir], dim=1)
            A = self.generate_adjacency(inputs, simi='cosine+QKV')
            x1, outputs = self.GAT_Branch(inputs, A=A)
            B1, WH1, C1 = x1.shape
            GAT_level1 = x1[:, 0:int(WH1 / 2), :] + x1[:, int(WH1 / 2):, :]  # [16,256,128]
            GAT_level1 = self.token2feature(GAT_level1)  # [16, 128, 16, 16]
            B, WH, C = outputs.shape
            s3_feat = outputs[:, 0:int(WH/2), :] + outputs[:, int(WH/2):, :]
            s3_feat = s3_feat + self.projector1(x1_color + x1_ir)
        s3_feat = self.token2feature(s3_feat)
        pred_boxes_stage1, score_lt_stage1, score_br_stage1 = self.head(s3_feat)
        # score_map_ctr, pred_boxes_stage1, size_map, offset_map = self.box_head(s3_feat)
        # lamb = 0.1
        # s3_feat = s3_feat * (1 - lamb)
        s1_feat = self.lddiffuse_x.predict(self.token2feature(self.projector2(x1_color)), self.token2feature(self.projector2(x1_ir)), GAT_level1)# * lamb
        # s0_feat = self.lddiffuse_z.predict(self.token2feature(z_color), self.token2feature(z_ir))
        # s1_feat = self.feature2token(s1_feat)
        pred_boxes_stage2, score_lt_stage2, score_br_stage2 = self.head_diffuse(s1_feat)

        pred_dict = dict()
        score_stage1 = score_lt_stage1.max().item() * score_br_stage1.max().item()
        score_stage2 = score_lt_stage2.max().item() * score_br_stage2.max().item()
        if score_stage1 > score_stage2:
           pred_boxes = pred_boxes_stage1
           pred_dict['score'] = score_stage1
        else:
           pred_boxes = pred_boxes_stage2
           pred_dict['score'] = score_stage2
        # pred_boxes = pred_boxes_stage1
        # pred_dict['score'] = score_stage1
        if self.debug:
            if self.use_visdom:
                for index in range(mask0.shape[0]):

                    self.visdom.register(mask0[index].squeeze().view(16, 16), 'heatmap', 1,
                                         'mask-{}'.format(time.strftime('%M:%S')))
                    self.visdom.register(score_center0[index].squeeze().view(16, 16), 'heatmap', 1,
                                         'score_center0-{}'.format(time.strftime('%M:%S')))
                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        # ----------- convert -------------------------
        outputs_coord = box_convert(pred_boxes, in_fmt='xyxy', out_fmt='cxcywh')

        pred_dict['box'] = outputs_coord.squeeze().detach().cpu().numpy()
        pred_dict['visualize'] = [None, None]

        return pred_dict

    def get_backbone_feature(self, z_color, z_ir, x1_color, x1_ir, x2_color=None, x2_ir=None, x3_color=None, x3_ir=None):
        # if self.cfg.backbone.type == 'ResNet':
        #     x1 = self.backbone(self._imagenet_norm(x1_color))[-1]
        #     z = self.backbone(self._imagenet_norm(z_color))[-1]
        #     x1 = x1.flatten(2).permute(0, 2, 1)
        #     z = z.flatten(2).permute(0, 2, 1)
        # else:
        if x2_color is not None:
            x2_color = self._imagenet_norm(x2_color)
            x2_ir = self._imagenet_norm(x2_ir)
            x3_color = self._imagenet_norm(x3_color)
            x3_ir = self._imagenet_norm(x3_ir)
            z, x1, x2, x3 = self.backbone(self._imagenet_norm(z_color), self._imagenet_norm(z_ir), self._imagenet_norm(x1_color), self._imagenet_norm(x1_ir), x2_color, x2_ir, x3_color, x3_ir)
            return z, x1, x2, x3
        else:
            z_color, x1_color, z_ir, x1_ir = self.backbone(self._imagenet_norm(z_color), self._imagenet_norm(z_ir),
                                      self._imagenet_norm(x1_color), self._imagenet_norm(x1_ir))
            return z_color, x1_color, z_ir, x1_ir

    def output_drop(self, output, target):
        delta1 = (output - target)**2
        # delta1 = self.criterion(output, self.target)
        batch_sz = output.shape[0]
        if self.eval() is True:
            batch_sz = batch_sz / 4
        delta = delta1.view(batch_sz, -1).sum(dim=1)
        sort_delta, index = torch.sort(delta, descending=True)
        for i in range(int(round(0.1 * batch_sz))):
            output[index[i], ...] = target[index[i], ...]
        return output


def build_translate_track(args):
    model = Model(args)
    return model

def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1] / 2)))
    d = x**2 + y**2
    g = np.exp(-0.5 / (sigma**2)*d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g.astype(np.float64)

class TrackerConfig(object):
    crop_sz = 224
    output_sz = 14

    lambda0 = 1e-4
    padding = 2.0
    output_sigma_factor = 0.1
    output_sigma = crop_sz / (1+padding)*output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, [output_sz, output_sz])
    # yf = torch.fft.rfft(torch.tensor(y).view(1, 1, output_sz, output_sz).cuda(), signal_dim=2)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from config.cfg_translation_track import cfg

    net = build_translate_track(cfg.model)

    gt = torch.Tensor([[0.1, 0.3, 0.7, 0.8], [0.4, 0.3, 0.7, 0.5]])
    x = torch.rand(2, 3, cfg.model.search_size[0], cfg.model.search_size[1])
    z = torch.rand(2, 3, cfg.model.template_size[0], cfg.model.template_size[1])

    in_dict = {
        'search': x,
        'template': z,
        'target': gt,
        'training': True,
    }

    out = net(in_dict)
    print(out)


    def prepare_input(resolution):
        input_dict = {
            'search': x,
            'template': z,
            'target': gt,
            'training': True,
        }

        return dict(input_dict=input_dict)


    flops, params = get_model_complexity_info(net,
                                              input_res=(None,),
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Macs:  ' + flops)
    print('      - Params: ' + params)
