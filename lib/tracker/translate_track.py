import cv2
import os
import torch
import numpy as np
from copy import deepcopy
from torchprofile import profile_macs
from ._tracker import Tracker


class TranslateT(Tracker):
    def __init__(self, hyper: dict, model):
        super(TranslateT, self).__init__()

        # updated hyper-params
        self.vis = False

        self.template_sf = None
        self.template_sz = None

        self.search_sf = None
        self.search_sz = None

        # --------------- hyper of this tracker
        self.score_threshold = None

        self.update_hyper_params(hyper)

        self.template_sz = self.template_sz[0]
        self.search_sz = self.search_sz[0]
        # ---------------
        self.model = model

        self.template_feat_sz = self.template_sz // self.model.backbone.out_stride
        self.search_feat_sz = self.search_sz // self.model.backbone.out_stride

        self.template_info = None

        self.language = None  # (N, L, 768)
        self.init_box = None  # [x y x y]
        self.last_box = None
        self.last_pos = None
        self.last_size = None
        self.last_score = None
        self.last_image = None

        self.imw = None
        self.imh = None
        self.channel_average = None

        self.idx = 0

    def init(self, im_color, im_ir, gt, **kwargs):  # BGRimg [x y w h]
        if self.vis:
            cv2.namedWindow('CommonTracker', cv2.WINDOW_NORMAL)

        self.set_deterministic()

        im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
        im_ir = cv2.cvtColor(im_ir, cv2.COLOR_BGR2RGB)
        x, y, w, h = gt

        self.idx = 1

        self.imh, self.imw = im_color.shape[:2]
        self.channel_average = np.mean(im_color, axis=(0, 1))

        self.init_box = np.array([x, y, x+w-1, y+h-1])
        self.last_box = np.array([x, y, x+w-1, y+h-1])
        self.last_score = 1
        self.last_image_color = np.array(im_color)
        self.last_image_ir = np.array(im_ir)

        self.last_pos = np.array([x+w/2, y+h/2])
        self.last_size = np.array([w, h])

        template_patch_color, template_patch_ir, template_roi, scale_f = self.crop_patch_fast(
            self.last_image_color, self.last_image_ir, self.init_box, scale_factor=self.template_sf, out_size=self.template_sz,
        )
        template_color, template_ir, boxes, _ = self.crop_patch_fast(
            self.last_image_color, self.last_image_ir, self.init_box, scale_factor=self.search_sf, out_size=self.search_sz,
        )
        test_outbox = False
        if test_outbox:
            im_show = cv2.cvtColor(template_patch_color, cv2.COLOR_RGB2BGR)
            cv2.rectangle(im_show, (int(template_roi[0]), int(template_roi[1])), (int(template_roi[2]), int(template_roi[3])), (0, 255, 0),
                          3)
            cv2.imshow('search_outbox', im_show)
            cv2.waitKey()
        self.template_info_color = self.to_pytorch(template_patch_color)
        self.template_info_ir = self.to_pytorch(template_patch_ir)
        self.template_info_color_s = self.to_pytorch(template_color)
        self.template_info_ir_s = self.to_pytorch(template_ir)
        self.boxes = torch.Tensor(boxes / 256).unsqueeze(0).cuda()

    def track(self, im_color, im_ir, **kwargs):
        # im_color = im_color.cpu().numpy()
        # im_color = im_color.cuda()
        # im_ir = im_ir.cpu().numpy()
        # im_ir = im_ir.cuda()
        curr_image_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
        curr_image_ir = cv2.cvtColor(im_ir, cv2.COLOR_BGR2RGB)
        # self.model.eval()
        # for param in self.model.parameters():
        #     param.requires_grad_(False)
        self.idx += 1

        curr_patch_color, curr_patch_ir, last_roi, scale_f = self.crop_patch_fast(
            curr_image_color, curr_image_ir, self.last_box, scale_factor=self.search_sf, out_size=self.search_sz,
        )
        self.last_roi = torch.Tensor(last_roi / 256).unsqueeze(0).cuda()
        with torch.no_grad():
            # img_color = torch.randn(288, 384, 3).cuda()
            # img_ir = torch.randn(288, 384, 3).cuda()
            # self.model.eval()
            # self.model = self.model.cpu()
            # for param in self.model.parameters():
            #     param.requires_grad_(False)
            # macs = profile_macs(self.model.track, (self.to_pytorch(curr_patch_color).cpu(), self.to_pytorch(curr_patch_ir).cpu(), self.template_info_color.cpu(), self.template_info_ir.cpu(), self.template_info_color_s.cpu(), self.template_info_ir_s.cpu(), self.boxes.cpu(), self.last_roi.cpu()))
            # flops = (2 * macs) / 1e9
            # print(f"GFLOPS: {flops:,}")
            pred_dict = self.model.track(self.to_pytorch(curr_patch_color), self.to_pytorch(curr_patch_ir), self.template_info_color, self.template_info_ir, self.template_info_color_s, self.template_info_ir_s, self.boxes, self.last_roi)
        pred_box = pred_dict['box']
        pred_score = pred_dict['score']
        self.last_roi = pred_box
        out_box, out_score = self.update_state(pred_box, pred_score, scale_f)

        #template update
        # bbox = np.array(out_box).astype(int)
        # bbox[2:] = bbox[2:] + bbox[:2] - 1
        # template_patch_color, template_patch_ir, template_roi, scale_f = self.crop_patch_fast(
        #     curr_image_color, curr_image_ir, bbox, scale_factor=self.template_sf,
        #     out_size=self.template_sz,
        # )
        # self.template_info_color = self.to_pytorch(template_patch_color)
        # self.template_info_ir = self.to_pytorch(template_patch_ir)
        # else:
        #     template_patch_color, template_patch_ir, template_roi, scale_f = self.crop_patch_fast(
        #         self.last_image_color, self.last_image_ir, self.init_box, scale_factor=self.template_sf,
        #         out_size=self.template_sz,
        #     )
        #     self.template_info_color = self.to_pytorch(template_patch_color)
        #     self.template_info_ir = self.to_pytorch(template_patch_ir)

        if self.vis:
            bb = np.array(out_box).astype(int)
            bb[2:] = bb[2:] + bb[:2] - 1
            im = cv2.rectangle(im_color, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 4)
            cv2.putText(im, '{:.2f}'.format(out_score), (40, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            cv2.imshow('CommonTracker', im)
            cv2.waitKey(1)

        return out_box, out_score, pred_dict['visualize']  # [x y w h]

