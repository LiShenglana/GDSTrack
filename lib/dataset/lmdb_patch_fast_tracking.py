import json
import math
import os

import cv2
import random
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import lmdb
import albumentations as aug
from copy import deepcopy
import imgaug.augmenters as iaa
import time
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from os.path import join
from lib.dataset.image_utils import *
sample_random = random.Random()

import torch
from lib.dataset._dataset import SubSet, BaseDataset
# from lib.dataset.crop_image_visualize import crop_image_visualize
from lib.dataset.generate_random_bbox import generate_bbox


"""
Examples:

tmp = {
    'data_set': data_set,
    'name'
    'num'
    'length'
}
data_set = [
    [
        {'path':[images/airplane-10/img/00000001.jpg], 
         'bbox':[x y w h], 
         'size':[im_w, im_h], 
         'name': lasot, ...
         }, 
         ...
    ],
    ...
]
"""

def lmdb_patchFT_collate_fn(batch):
    template_img_color = [torch.Tensor(item[0]).unsqueeze(0) for item in batch]
    template_img_ir = [torch.Tensor(item[1]).unsqueeze(0) for item in batch]
    s3_box = [torch.Tensor(item[2]).unsqueeze(0) for item in batch]
    search3_img_color = [torch.Tensor(item[3]).unsqueeze(0) for item in batch]
    search3_img_ir = [torch.Tensor(item[4]).unsqueeze(0) for item in batch]
    s_box = [torch.Tensor(item[5]).unsqueeze(0) for item in batch]
    search_img_color = [torch.Tensor(item[6]).unsqueeze(0) for item in batch]
    search_img_ir = [torch.Tensor(item[7]).unsqueeze(0) for item in batch]

    template_img_color = torch.cat(template_img_color, dim=0)
    template_img_ir = torch.cat(template_img_ir, dim=0)
    search3_img_color = torch.cat(search3_img_color, dim=0)
    search3_img_ir = torch.cat(search3_img_ir, dim=0)
    s3_box = torch.cat(s3_box, dim=0)
    search_img_color = torch.cat(search_img_color, dim=0)
    search_img_ir = torch.cat(search_img_ir, dim=0)
    s_box = torch.cat(s_box, dim=0)

    return {
        'template_color': template_img_color,
        'template_ir': template_img_ir,
        's3_box': s3_box,
        'search3_color': search3_img_color,
        'search3_ir': search3_img_ir,
        'template_ori': s_box,
        'search_color': search_img_color,
        'search_ir': search_img_ir,
    }


class LMDBPatchFastTracking(BaseDataset):

    def __init__(self, cfg, lmdb_path, json_path, dataset_name_list: list = None, num_samples: int = None):
        super(LMDBPatchFastTracking).__init__()

        self.debug = False
        if self.debug:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)

        self.lmdb_path: dict = lmdb_path
        self.json_path: dict = json_path

        self.sample_range: int = cfg.sample_range

        self.search_sz: List[int, int] = cfg.search_size
        self.search_scale_f: float = cfg.search_scale_f
        self.search_jitter_f: List[float, float] = cfg.search_jitter_f

        self.template_sz: List[int, int] = cfg.template_size
        self.template_scale_f: float = cfg.template_scale_f
        self.template_jitter_f: List[float, float] = cfg.template_jitter_f

        # Declare an augmentation pipeline
        self.aug = aug.Compose([
            aug.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=1),
            # aug.ToGray(p=0.05),
            # aug.CoarseDropout(),
            aug.GaussianBlur(blur_limit=0, sigma_limit=(0.2, 1), p=0.5),
            # aug.HorizontalFlip(p=0.5),
        ])

        # Augmentation for template patch
        self.template_aug_seq = iaa.Sequential([
            # iaa.Fliplr(0.4),
            # iaa.Flipud(0.2),
            # iaa.PerspectiveTransform(scale=(0.01, 0.07)),
            iaa.CoarseDropout((0.0, 0.05), size_percent=0.15, per_channel=0.5),
            iaa.SaltAndPepper(0.05, per_channel=True),
        ], random_order=True)

        # Augmentation for search area
        self.search_aug_seq1 = iaa.Sequential([
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.MotionBlur(k=(3, 9), angle=[-60, 60]),
        ], random_order=True)

        self.search_aug_seq2 = iaa.Sequential([
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.MotionBlur(k=(3, 9), angle=[-60, 60]),
        ], random_order=True)

        self.search_aug_seq3 = iaa.Sequential([
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.MotionBlur(k=(3, 9), angle=[-60, 60]),
        ], random_order=True)

        # Response map size
        self.size = 9
        # Feature size of template patch
        self.tf_size = 8
        # Feature axis of search area (designed to be the same as response map size in USOT v1)
        self.sf_size = 16
        # Total stride of backbone
        self.stride = 16
        self.id = 1
        # Response map grid
        sz = self.size
        sz_x = sz // 2
        sz_y = sz // 2
        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))
        self.grid_to_search = {}
        self.grid_to_search_x = x * 16 + 256 // 2
        self.grid_to_search_y = y * 16 + 256 // 2
        self.mean = np.expand_dims(np.expand_dims(np.array([109, 120, 119]), axis=0), axis=0).astype(np.float32)

        self.train_datas = []
        self.video_quality = cfg.VIDEO_QUALITY
        self.memory_num = cfg.MEMORY_NUM
        self.far_sample = cfg.FAR_SAMPLE
        start = 0
        self.num = 0
        for data_name in dataset_name_list:
            dataset = subData(cfg, data_name, start, self.memory_num,
                              self.video_quality, self.far_sample)
            self.train_datas.append(dataset)
            # Real video number
            start += dataset.num
            # The number used for subset shuffling
            self.num += dataset.num_use

        self._shuffle()

        # self.LMDB_ENVS = {}
        # self.LMDB_HANDLES = {}
        # self.dataset_list: List = []
        # for name in dataset_name_list:
        #     dataset = SubSet(name=name, load=self.json_path[name])
        #     self.dataset_list.append([dataset.data_set, len(dataset.data_set)])
        #
        #     env = lmdb.open(self.lmdb_path[name], readonly=True, lock=False, readahead=False, meminit=False)
        #     self.LMDB_ENVS[name] = env
        #     item = env.begin(write=False)
        #     # for key, value in item.cursor():
        #     #     print(key, value)
        #     self.LMDB_HANDLES[name] = item
        # self.num_samples = num_samples

    def _clean(self):
        """
        Remove empty videos/frames/annos in dataset
        """
        # No frames
        to_del = []
        for video in self.labels:
            frames = self.labels[video]
            if len(frames) <= 0:
                print("warning {} has no frames.".format(video))
                to_del.append(video)

        for video in to_del:
            try:
                del self.labels[video]
            except:
                pass

    def __len__(self):
        return self.num

    def __getitem__(self, item):

        index = self.pick[item]
        dataset, index = self._choose_dataset(index)
        pair_info = dataset._get_instances(index=index, cycle_memory=True)
        #
        # Here only one template frame image is returned, and it will be utilized for both template and search
        _search1_img_color = cv2.imread(pair_info[0])
        _search1_img_ir = cv2.imread(pair_info[1])
        _s1_box = pair_info[2] #[x1, y1, x2, y2]
        _template_img_color = _search1_img_color
        _template_img_ir = _search1_img_ir
        _t_box = _s1_box
        search_images_nearby_color = [cv2.imread(image_path) for image_path in pair_info[3]]
        search_images_nearby_ir = [cv2.imread(image_path) for image_path in pair_info[4]]
        search_bbox_nearby = [pair_info[5][i] for i in range(len(search_images_nearby_color))]

        # video_list, list_len = random.choice(self.dataset_list)
        # idx = np.random.randint(0, list_len)
        #
        # t_dict, s1_dict, s2_dict = self.check_sample(video_list[idx], video_list, self.sample_range)
        #
        # # read RGB image, [x y w h]
        # _template_img_color, _template_img_ir, _t_box, t_lang = self.parse_frame_lmdb(t_dict, self.LMDB_HANDLES, need_language=True)
        # _search1_img_color, _search1_img_ir, _s1_box, s1_lang = self.parse_frame_lmdb(s1_dict, self.LMDB_HANDLES, need_language=True)
        # _search2_img_color, _search2_img_ir, _s2_box, s2_lang = self.parse_frame_lmdb(s2_dict, self.LMDB_HANDLES,
        #                                                                               need_language=True)
        #
        # generator_bbox = generate_bbox()

        # _t_box = generator_bbox.generate(_template_img_color, self.id)

        # _t_box = generate(_template_img_color, _template_img_ir)
        # _t_box = sam_generate(_template_img_color)
        # if _t_box is None:
        #     _t_box = generator_bbox.generate(_template_img_color)

        # self.id = self.id + 1
        # _s1_box = deepcopy(_t_box)
        # _s2_box = deepcopy(_t_box)
        # _s3_box = deepcopy(_t_box)
        #
        # crop_image_visualize(_template_img_color, _template_img_ir, _t_box, _search1_img_color, _search1_img_ir, _s1_box,
        #                      _search2_img_color, _search2_img_ir, _s2_box, _search2_img_color, _search2_img_ir, _s3_box)
        t_box_copy = deepcopy(np.array(_t_box))
        s1_box_copy = deepcopy(np.array(search_bbox_nearby[0]))
        s2_box_copy = deepcopy(np.array(search_bbox_nearby[1]))
        t_box_copy[2:] = t_box_copy[2:] - t_box_copy[:2] + 1
        s1_box_copy[2:] = s1_box_copy[2:] - s1_box_copy[:2] + 1
        s2_box_copy[2:] = s2_box_copy[2:] - s2_box_copy[:2] + 1

        template_img_color, template_img_ir, _, t_box, shift_t = self.crop_patch_fast(
            _template_img_color, _template_img_ir, t_box_copy,
            out_size=self.template_sz, padding=0, #out_size=self.template_sz
            scale_factor=self.template_scale_f, #self.template_scale_f,
            jitter_f=self.template_jitter_f) #self.template_jitter_f

        search_img_color, search_img_ir, _, s_box, shift_s = self.crop_patch_fast(
            search_images_nearby_color[0], search_images_nearby_ir[0], s1_box_copy,
            out_size=self.search_sz, padding=0,
            scale_factor=self.search_scale_f,#self.search_scale_f,
            jitter_f=self.search_jitter_f)

        search3_img_color, search3_img_ir, _, s3_box, shift_s3 = self.crop_patch_fast(
            search_images_nearby_color[1], search_images_nearby_ir[1], s2_box_copy,
            out_size=self.search_sz, padding=0,
            scale_factor=self.search_scale_f,  # self.search_scale_f,
            jitter_f=self.search_jitter_f)

        # template_img_color = template_img_color - self.mean
        # search1_img_color = search1_img_color - self.mean
        # search2_img_color = search2_img_color - self.mean
        # search3_img_color = search3_img_color - self.mean

        test_outbox = False
        if test_outbox:
            im_show1 = cv2.cvtColor(search_img_color, cv2.COLOR_RGB2BGR)
            cv2.rectangle(im_show1, (int(s_box[0]), int(s_box[1])),
                          (int(s_box[0]) + int(s_box[2]), int(s_box[1]) + int(s_box[3])), (0, 255, 0),
                          3)
            im_show2 = cv2.cvtColor(search3_img_color, cv2.COLOR_RGB2BGR)
            cv2.rectangle(im_show2, (int(s3_box[0]), int(s3_box[1])),
                          (int(s3_box[0]) + int(s3_box[2]), int(s3_box[1]) + int(s3_box[3])), (0, 255, 0),
                          3)
            cv2.imshow('template_img_color', im_show1)
            cv2.imshow('search1_img_color', im_show2)
            cv2.waitKey()

        #
        # self.calchist_for_rgb(search3_img_color, s3_box, self.id)
        # self.id = self.id + 1

        # crop_image_visualize(template_img_color, template_img_ir, t_box, search1_img_color, search1_img_ir,
        #                      s1_box, search2_img_color, search2_img_ir, s2_box, search3_img_color, search3_img_ir,
        #                      s3_box, self.id)
        # self.id = self.id + 1

        # template_img_color, template_img_ir, search_img_color, search_img_ir = self._augmentation(template_img_color, template_img_ir, search_img_color, search_img_ir)
        template_img_color, template_img_ir, search_img_color, search_img_ir, search3_img_color, search3_img_ir = map(lambda im: self.aug(image=im)["image"], [template_img_color, template_img_ir, search_img_color, search_img_ir, search3_img_color, search3_img_ir])
        #
        a = np.random.rand(2, 1)
        #
        # s1_box_ori = deepcopy(s1_box)
        # s2_box_ori = deepcopy(s2_box)
        # s_box_ori = deepcopy(s_box)
        #
        # if a[0] < 0.5:
        #     search1_img_color, s1_box = self.horizontal_flip(search1_img_color, s1_box_ori)
        #     search1_img_ir = self.horizontal_flip(search1_img_ir)
        #
        # if a[1] < 0.5:
        #     search2_img_color, s2_box = self.horizontal_flip(search2_img_color, s2_box_ori)
        #     search2_img_ir = self.horizontal_flip(search2_img_ir)
        #
        if a[0] < 0.5:
            search_img_color, s_box = self.horizontal_flip(search_img_color, s_box)
            search_img_ir = self.horizontal_flip(search_img_ir)
        if a[1] < 0.5:
            search3_img_color, s3_box = self.horizontal_flip(search3_img_color, s3_box)
            search3_img_ir = self.horizontal_flip(search3_img_ir)

        # crop_image_visualize(template_img_color, template_img_ir, t_box, search1_img_color, search1_img_ir, s1_box, search2_img_color, search2_img_ir,
        #                      s2_box, search3_img_color, search3_img_ir, s3_box)

        # crop_image_visualize(search1_img_color, search1_img_ir, s1_box, search2_img_color, search2_img_ir,
        #                          s2_box, search3_img_color, search3_img_ir, s3_box, search3_img_color, search3_img_ir,
        #                          s3_box)

        # ori_center = np.array([float(self.search_sz[0] / 2), float(self.search_sz[0] / 2)])  # 112 - 1
        # out_center1 = s1_box[0:2] + 0.5 * s1_box[2:4]
        # shift_s1 = out_center1 - ori_center
        # out_center2 = s2_box[0:2] + 0.5 * s2_box[2:4]
        # shift_s2 = out_center2 - ori_center
        # out_center3 = s3_box[0:2] + 0.5 * s3_box[2:4]
        # shift_s3 = out_center3 - ori_center

        # if 'left' in s_lang:
            #     s_lang = s_lang.replace('left', 'right')
            # elif 'right' in s_lang:
            #     s_lang = s_lang.replace('right', 'left')

        # cv2.imwrite("/home/cscv/Documents/lsl/MATPrompt/lib/dataset/search.jpg", search1_img_color)
        # cv2.imwrite("/home/cscv/Documents/lsl/MATPrompt/lib/dataset/template.jpg", template_img_color)

        # if self.debug:
        #     print(t_box.astype(int), s1_box.astype(int), s1_lang)
        #     self.debug_fn([template_img_color, search1_img_color], [t_box, s1_box])
        #     self.debug_fn([template_img_ir, search1_img_ir], [t_box, s1_box])


        template_img_color, search_img_color, search3_img_color = map(lambda x: x.transpose(2, 0, 1).astype(np.float64), [template_img_color, search_img_color, search3_img_color])
        template_img_ir, search_img_ir, search3_img_ir = map(lambda x: x.transpose(2, 0, 1).astype(np.float64),
                                       [template_img_ir, search_img_ir, search3_img_ir])
        t_box, s_box, s3_box = map(lambda x: x.astype(np.float64), [t_box, s_box, s3_box])  # [x, y, w, h]

        # out_label_t = self._create_labels([9, 9], shift_s3)
        # out_label_s1 = self._create_labels([16, 16], shift_s1)
        # out_label_s2 = self._create_labels([16, 16], shift_s2)
        # [x, y, w, h] -> norm[x, y, x, y]
        # t_box[2:] = t_box[:2] + t_box[2:] - 1
        # t_box[0::2] = t_box[0::2] / self.search_sz[1]
        # t_box[1::2] = t_box[1::2] / self.search_sz[0]
        #
        # s1_box[2:] = s1_box[:2] + s1_box[2:] - 1
        # s1_box[0::2] = s1_box[0::2] / self.search_sz[1]
        # s1_box[1::2] = s1_box[1::2] / self.search_sz[0]
        #
        # s2_box[2:] = s2_box[:2] + s2_box[2:] - 1
        # s2_box[0::2] = s2_box[0::2] / self.search_sz[1]
        # s2_box[1::2] = s2_box[1::2] / self.search_sz[0]

        s_box[2:] = s_box[:2] + s_box[2:] - 1
        # reg_label, reg_weight = self.reg_label(s3_box)
        s_box[0::2] = s_box[0::2] / self.search_sz[1]
        s_box[1::2] = s_box[1::2] / self.search_sz[0]

        s3_box[2:] = s3_box[:2] + s3_box[2:] - 1
        s3_box[0::2] = s3_box[0::2] / self.search_sz[1]
        s3_box[1::2] = s3_box[1::2] / self.search_sz[0]

        return template_img_color, template_img_ir, s3_box, search3_img_color, search3_img_ir, s_box, search_img_color, search_img_ir

    def _shuffle(self):
        """
        Random shuffle
        """
        pick = []
        m = 0
        while m < self.num:
            p = []
            for subset in self.train_datas:
                sub_p = subset.pick
                p += sub_p
            sample_random.shuffle(p)

            pick += p
            m = len(pick)
        self.pick = pick
        print("dataset length {}".format(self.num))

    def _toBBox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.template_sz[0]

        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def _choose_dataset(self, index):
        for dataset in self.train_datas:
            if dataset.start + dataset.num > index:
                return dataset, index - dataset.start



    def calchist_for_rgb(self, img, s3_box, id):
        path = "/home/cscv/Documents/lsl/MATPrompt (latefusion)/hist-vis/test/id-{}".format(id)
        if not os.path.exists(path):
            os.mkdir(path)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # mean_value = int(np.mean(img1))
        # img1 = img1 - mean_value
        # normal_img = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
        box = s3_box
        box[2:4] = box[0:2] + box[2:4]
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_img = img1[y1:y2, x1:x2]
        variance = str(np.var(cropped_img))
        histb = cv2.calcHist([cropped_img], [0], None, [256], [0, 256])
        # histg = cv2.calcHist([img1], [0], None, [256], [0, 256])
        # histr = cv2.calcHist([cropped_img], [2], None, [256], [0, 255])

        plt.plot(histb, color="b")
        # plt.plot(histg, color="g")
        # plt.plot(histr, color="r")
        plt.savefig(path + "/result_hist_RGB.jpg")
        plt.clf()

        im_show = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cropped_img = cv2.putText(cropped_img, variance, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        img = cv2.rectangle(im_show, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), (0, 255, 0),
                      3)
        # cv2.imshow('template_img_color', im_show)
        cv2.imwrite(path + '/template_img_color.jpg', img)

    def reg_label(self, bbox):
        """
        Generate regression label
        :param bbox: [x1, y1, x2, y2]
        :return: [l, t, r, b]
        """
        x1, y1, x2, y2 = bbox
        l = self.grid_to_search_x - x1
        t = self.grid_to_search_y - y1
        r = x2 - self.grid_to_search_x
        b = y2 - self.grid_to_search_y

        l, t, r, b = map(lambda x: np.expand_dims(x, axis=-1), [l, t, r, b])
        reg_label = np.concatenate((l, t, r, b), axis=-1)
        reg_label_min = np.min(reg_label, axis=-1)
        inds_nonzero = (reg_label_min > 0).astype(float)

        return reg_label, inds_nonzero

    def _augmentation(self, template_img_color, template_img_ir, search_img_color, search_img_ir, search=False, cycle_memory=False):
        """
        Data augmentation for input frames
        """
        # shape_temp = template_img_color.shape
        # shape_search = search1_img_color.shape
        # crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size, size))
        # param = edict()
        #
        # if not search:
        #     # The shift and scale for template
        #     param.shift = (self._posNegRandom() * self.shift, self._posNegRandom() * self.shift)  # shift
        #     param.scale = (
        #         (1.0 + self._posNegRandom() * self.scale), (1.0 + self._posNegRandom() * self.scale))  # scale change
        # elif not cycle_memory:
        #     # The shift and scale for search area
        #     param.shift = (self._posNegRandom() * self.shift_s, self._posNegRandom() * self.shift_s)  # shift
        #     param.scale = (
        #         (1.0 + self._posNegRandom() * self.scale_s),
        #         (1.0 + self._posNegRandom() * self.scale_s))  # scale change
        # else:
        #     # The shift and scale for memory search areas
        #     param.shift = (self._posNegRandom() * self.shift_m, self._posNegRandom() * self.shift_m)  # shift
        #     param.scale = (
        #         (1.0 + self._posNegRandom() * self.scale_m),
        #         (1.0 + self._posNegRandom() * self.scale_m))  # scale change
        #
        # crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)
        #
        # x1, y1 = crop_bbox.x1, crop_bbox.y1
        # bbox = BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)
        #
        # scale_x, scale_y = param.scale
        # bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)
        #
        # image_color = self._crop_hwc(image_color, crop_bbox, size)  # shift and scale
        # image_ir = self._crop_hwc(image_ir, crop_bbox, size)  # shift and scale
        # bbs = BoundingBoxesOnImage([
        #     BoundingBox(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2)],
        #     shape=image_color.shape)

        # images =[image_color, image_ir]
        template_images = torch.cat((torch.tensor(template_img_color).unsqueeze(0), torch.tensor(template_img_ir).unsqueeze(0)), 0).numpy()  #[2,128,128,3]
        template_images = self.template_aug_seq(images=template_images)

        search_images3 = torch.cat(
            (torch.tensor(search_img_color).unsqueeze(0), torch.tensor(search_img_ir).unsqueeze(0)),
            0).numpy()  # [2,128,128,3]
        search_images3 = self.search_aug_seq3(images=search_images3)
        # ia.imshow(np.hstack([bbs_aug.draw_on_image(images[0]), bbs_aug.draw_on_image(images[1])]))
        # image_ir, _ = self.template_aug_seq(image=image_ir, bounding_boxes=bbs)
        # image_color = torch.chunk(torch.tensor(image_cat), 2, 2)[0].numpy()
        # image_ir = torch.chunk(torch.tensor(image_cat), 2, 2)[1].numpy()

        # bbox = Corner(self.clip_number(bbs_aug[0].x1, _max=images[0].shape[0]),
        #               self.clip_number(bbs_aug[0].y1, _max=images[0].shape[1]),
        #               self.clip_number(bbs_aug[0].x2, _max=images[0].shape[0]),
        #               self.clip_number(bbs_aug[0].y2, _max=images[0].shape[1]))

        return template_images[0], template_images[1], search_images3[0], search_images3[1]#, bbox, param

    def _calc_video_quality(self, bbox_picked_freq, corner_bbox_freq):
        """
        The function to calculate video quality with DP-selection frequency in video
        In practice, we additionally give penalty to video sequences with lots of pseudo boxes lying at the corner
        """

        return bbox_picked_freq - 1 / 3 * corner_bbox_freq

    def _calc_short_term_frame_quality(self, bbox_info):
        """
        The function to calculate short-term frame quality for sampling template frames for naive Siamese tracker
        bbox_info structure:
             index 0-3: [x1, y1, x2, y2]
             index 4-5: [short-term DP-pick-freq, long-term DP-pick-freq] for a single frame
             index 6-8: [T_l, T_u, corner_score]
        We use short-term DP-pick-freq as the basic frame quality, and use corner_score to refine it.
        As an implementation detail, we give penalty to frames with their pseudo boxes lying at the corner
        """

        return bbox_info[4] + 2 / 3 * bbox_info[8]

    def _calc_long_term_frame_quality(self, bbox_info, video_len):
        """
        The function to calculate long-term frame quality for sampling template frames for cycle memory training
        bbox_info structure:
             index 0-3: [x1, y1, x2, y2]
             index 4-5: [short-term DP-pick-freq, long-term DP-pick-freq] for a single frame
             index 6-8: [T_l, T_u, corner_score]
        We use short-term DP-pick-freq as the basic frame quality, and use T_u, T_l, and corner_score to refine it.
        As an implementation detail, we give penalty to frames with their pseudo boxes lying at the corner.
        For cycle memory, we also give priority to template frames with higher (T_u - T_l) frame intervals
        """

        return bbox_info[4] + 1 / 2 * bbox_info[8] + (bbox_info[7] - bbox_info[6]) / (video_len * 2)

    def _create_labels(self, size, shift, r_pos=2, r_neg=0):
        # if hasattr(self, 'label') and self.label.size() == size:
        #     return self.label

        def logistic_labels(x, y, r_pos=2, r_neg=0):
            dist = np.abs(x) + np.abs(y)
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        h, w = size
        if math.isnan(shift[0] / 16):
            x = np.arange(w) - np.floor(float(w / 2))
        else:
            x = np.arange(w) - np.floor(float(w / 2 + int(shift[0] / 16)))
        if math.isnan(shift[1] / 16):
            y = np.arange(h) - np.floor(float(h / 2))
        else:
            y = np.arange(h) - np.floor(float(h / 2 + int(shift[1] / 16)))
        x, y = np.meshgrid(x, y)

        label = logistic_labels(x, y, r_pos, r_neg)
        return label


    def debug_fn(self, im, box):  # [x, y, x, y]
        t_img = im[0]
        s_img = im[1]

        t_bbox = box[0]
        s_bbox = box[1]

        # t_img = cv2.rectangle(
        #     t_img,
        #     (int(t_bbox[0]), int(t_bbox[1])),
        #     (int(t_bbox[0] + t_bbox[2] - 1), int(t_bbox[1] + t_bbox[3] - 1)), (0, 255, 0), 4)
        #
        # s_img = cv2.rectangle(
        #     s_img,
        #     (int(s_bbox[0]), int(s_bbox[1])),
        #     (int(s_bbox[0] + s_bbox[2] - 1), int(s_bbox[1] + s_bbox[3] - 1)), (0, 255, 0), 4)

        t_img = cv2.rectangle(
            t_img,
            (int(t_bbox[0]), int(t_bbox[1])),
            (int(t_bbox[2] - 1), int(t_bbox[3] - 1)), (0, 255, 0), 4)

        s_img = cv2.rectangle(
            s_img,
            (int(t_bbox[0]), int(t_bbox[1])),
            (int(t_bbox[2] - 1), int(t_bbox[3] - 1)), (0, 255, 0), 4)

        self.ax1.imshow(t_img)
        self.ax2.imshow(s_img)
        self.fig.show()
        plt.waitforbuttonpress()

class subData(object):
    """
    Sub dataset class for training USOT with multi dataset
    """

    def __init__(self, cfg, data_name, start, memory_num, video_quality, far_sample):
        self.data_name = data_name
        self.start = start

        # Dataset info
        # info = cfg.USOT.DATASET[data_name]
        self.root = cfg.path

        with open(cfg.annotation) as fin:
            self.labels = json.load(fin)
            self._clean()
            # Video number
            self.num = len(self.labels)

        # Number of training instances used in each epoch for a certain dataset
        self.num_use = cfg.use
        # Number of memory frames in a single training instance
        self.memory_num = memory_num
        # The threshold to filter videos
        self.video_quality = video_quality
        # When sampling memory frames, first sample (memory_num + far_sample) frames in the video fragment,
        #             and then pick (memory_num) frames "most far from" the template frame
        self.far_sample = far_sample

        self._shuffle()

    def _clean(self):
        """
        Remove empty videos/frames/annos in dataset
        """
        # No frames
        to_del = []
        for video in self.labels:
            frames = self.labels[video]
            if len(frames) <= 0:
                print("warning {} has no frames.".format(video))
                to_del.append(video)

        for video in to_del:
            try:
                del self.labels[video]
            except:
                pass

        print(self.data_name)

        self.videos = list(self.labels.keys())
        print('{} loaded.'.format(self.data_name))

    def _shuffle(self):
        """
        Shuffle to get random pairs index (video)
        """
        lists = list(range(self.start, self.start + self.num))
        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]
        return self.pick

    def _calc_video_quality(self, bbox_picked_freq, corner_bbox_freq):
        """
        The function to calculate video quality with DP-selection frequency in video
        In practice, we additionally give penalty to video sequences with lots of pseudo boxes lying at the corner
        """

        return bbox_picked_freq - 1 / 3 * corner_bbox_freq

    def _calc_short_term_frame_quality(self, bbox_info):
        """
        The function to calculate short-term frame quality for sampling template frames for naive Siamese tracker
        bbox_info structure:
             index 0-3: [x1, y1, x2, y2]
             index 4-5: [short-term DP-pick-freq, long-term DP-pick-freq] for a single frame
             index 6-8: [T_l, T_u, corner_score]
        We use short-term DP-pick-freq as the basic frame quality, and use corner_score to refine it.
        As an implementation detail, we give penalty to frames with their pseudo boxes lying at the corner
        """

        return bbox_info[4] + 2 / 3 * bbox_info[8]

    def _calc_long_term_frame_quality(self, bbox_info, video_len):
        """
        The function to calculate long-term frame quality for sampling template frames for cycle memory training
        bbox_info structure:
             index 0-3: [x1, y1, x2, y2]
             index 4-5: [short-term DP-pick-freq, long-term DP-pick-freq] for a single frame
             index 6-8: [T_l, T_u, corner_score]
        We use short-term DP-pick-freq as the basic frame quality, and use T_u, T_l, and corner_score to refine it.
        As an implementation detail, we give penalty to frames with their pseudo boxes lying at the corner.
        For cycle memory, we also give priority to template frames with higher (T_u - T_l) frame intervals
        """

        return bbox_info[4] + 1 / 2 * bbox_info[8] + (bbox_info[7] - bbox_info[6]) / (video_len * 2)

    def _get_instances(self, index=0, cycle_memory=False):
        """
        get training instances
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track_id1 = random.choice(list(video.keys()))

        if cycle_memory:
            # For cycle memory training (returning one search frame and several memory frames)
            return self._get_cycle_memory_image_anno(video_name, track_id1, video_index=index)
        else:
            # For offline naive Siamese tracker (one template and one search area picked in the same frame)
            return self._get_siamese_image_anno(video_name, track_id1, video_index=index)

    def _get_siamese_image_anno_groundTruth(self, video, track_id, video_index=None):
        """
        Loader logic for naive Siamese training
        Sampling the template frame and obtaining its pseudo annotation
        """
        video_info = self.labels[video]
        track_info = video_info[track_id]

        # Threshold to pick reliable videos

        # Sample more candidate frames if the video quality is lower, and vice versa
        frame_candidate_num = 3

        # Branch 1
        # If the picked video is of high quality to be used as training data, simply pick the most reliable frame
        frames = list(track_info.keys())
        if 'meta' in frames:
            frames.remove('meta')
        video_len = len(frames)
        picked_frame_candidates_s = np.random.choice(video_len, frame_candidate_num, replace=True)

        max_cand_frame_s = np.random.choice(picked_frame_candidates_s)

        frame_id_s = frames[int(max_cand_frame_s)]
        image_path_s_color = join(self.root, video, 'visible',
                                  "{}.{}.x.jpg".format(frame_id_s, track_id))
        image_path_s_ir = join(self.root, video, 'infrared', "{}.{}.x.jpg".format(frame_id_s, track_id))
        # Return the single frame for template-search pair
        return image_path_s_color, image_path_s_ir, track_info[frame_id_s][:4]
            #return image_path_s_color, track_info[frame_id_s][:4]

    def _get_siamese_image_anno(self, video, track_id, video_index=None):
        """
        Loader logic for naive Siamese training
        Sampling the template frame and obtaining its pseudo annotation
        """
        video_info = self.labels[video]
        track_info = video_info[track_id]

        # Threshold to pick reliable videos
        video_tolerance_threshold = self.video_quality

        # Step 1: testify the currently picked video by video quality score
        bbox_picked_freq = track_info['meta']['bbox_picked_freq']
        corner_bbox_freq = track_info['meta']['corner_bbox_freq']
        video_quality_score = self._calc_video_quality(bbox_picked_freq, corner_bbox_freq)

        # Sample more candidate frames if the video quality is lower, and vice versa
        frame_candidate_num = int((1.0 / bbox_picked_freq) * 3)

        # Branch 1
        # If the picked video is of high quality to be used as training data, simply pick the most reliable frame
        if video_quality_score >= video_tolerance_threshold and corner_bbox_freq < 0.25:
            frames = list(track_info.keys())
            if 'meta' in frames:
                frames.remove('meta')
            video_len = len(frames)
            picked_frame_candidates_s = np.random.choice(video_len, frame_candidate_num, replace=True)

            # Calculate short-term frame quality
            short_term_frame_quality_s = np.array([self._calc_short_term_frame_quality(track_info[frames[cand]])
                                                   for cand in picked_frame_candidates_s])
            # Select the frame with the highest frame quality
            max_cand_index_s = np.argmax(short_term_frame_quality_s)
            max_cand_frame_s = picked_frame_candidates_s[max_cand_index_s]

            frame_id_s = frames[int(max_cand_frame_s)]
            image_path_s_color = join(self.root, video, 'visible',
                                          "{}.{}.x.jpg".format(frame_id_s, track_id))
            image_path_s_ir = join(self.root, video, 'infrared', "{}.{}.x.jpg".format(frame_id_s, track_id))
            # Return the single frame for template-search pair
            return image_path_s_color, image_path_s_ir, track_info[frame_id_s][:4]
            #return image_path_s_color, track_info[frame_id_s][:4]

        # Branch 2
        # If the picked video is not of high quality, re-sample video from its nearby videos
        # Step 2: re-sample video for the original randomly sampled video is of low quality
        video_total_num = len(self.labels)
        candidate_range = np.arange(max(0, video_index - 30), min(video_total_num - 1, video_index + 31))

        # Sample another video from nearby videos, and pick the video with the highest quality score
        max_pick_times = 20
        video_candidate_num = 3
        max_freq_video = None
        track_id = None
        while max_pick_times:
            picked_candidates = np.random.choice(candidate_range, video_candidate_num, replace=True)
            picked_candidates_video_name = [self.videos[cand] for cand in picked_candidates]
            picked_track_id = [random.choice(list(self.labels[video_name].keys()))
                               for video_name in picked_candidates_video_name]
            video_scores = np.array([self._calc_video_quality(
                self.labels[picked_candidates_video_name[cand_ind]][picked_track_id[cand_ind]]['meta'][
                    'bbox_picked_freq'],
                self.labels[picked_candidates_video_name[cand_ind]][picked_track_id[cand_ind]]['meta'][
                    'corner_bbox_freq'])
                for cand_ind in range(len(picked_candidates_video_name))])
            max_freq_index = np.argmax(video_scores)
            max_freq_video = picked_candidates[max_freq_index]
            track_id = picked_track_id[max_freq_index]

            # Check if the currently selected video is of high quality or not
            if video_scores[max_freq_index] > video_tolerance_threshold:
                break
            else:
                max_pick_times -= 1

        # Extreme case: if no video is determined even after 20 trials, then randomly pick one.
        if max_freq_video is None or track_id is None:
            max_freq_video = np.random.choice(candidate_range, 1)
            track_id = random.choice(list(self.labels[self.videos[max_freq_video]].keys()))

        # Re-sampling video finished
        video = self.videos[max_freq_video]
        video_info = self.labels[video]
        track_info = video_info[track_id]
        bbox_picked_freq = track_info['meta']['bbox_picked_freq']
        frame_candidate_num = int((1.0 / bbox_picked_freq) * 3)

        # Step 3: re-sample frames according to the frame quality
        frames = list(track_info.keys())
        if 'meta' in frames:
            frames.remove('meta')
        video_len = len(frames)
        picked_frame_candidates_s = np.random.choice(video_len, frame_candidate_num, replace=True)

        # Calculate short-term frame quality
        short_term_frame_quality_s = np.array([self._calc_short_term_frame_quality(track_info[frames[cand]])
                                               for cand in picked_frame_candidates_s])
        # Select the frame with the highest frame quality
        max_cand_index_s = np.argmax(short_term_frame_quality_s)
        max_cand_frame_s = picked_frame_candidates_s[max_cand_index_s]

        frame_id_s = frames[int(max_cand_frame_s)]
        frame_id_s_format = '0' * (8 - len(frame_id_s)) + frame_id_s
        image_path_s_color = join(self.root, video, 'visible', "{}.{}.x.jpg".format(frame_id_s, track_id))
        image_path_s_ir = join(self.root, video, 'infrared', "{}.{}.x.jpg".format(frame_id_s, track_id))

        # Return the single frame for template-search pair
        return image_path_s_color, image_path_s_ir, track_info[frame_id_s][:4]

    def _get_cycle_memory_image_anno(self, video, track_id, video_index=None):

        """
        Loader logic for cycle memory training
        Sampling the template frame (with pseudo annotation) as well as N_mem memory frames
        """
        video_info = self.labels[video]
        track_info = video_info[track_id]

        # Threshold to pick reliable videos
        video_tolerance_threshold = self.video_quality

        # Step 1: test the currently picked video
        bbox_picked_freq = track_info['meta']['bbox_picked_freq']
        corner_bbox_freq = track_info['meta']['corner_bbox_freq']
        video_quality_score = self._calc_video_quality(bbox_picked_freq, corner_bbox_freq)

        # Sample more candidate frames if the video quality is lower, and vice versa
        frame_candidate_num = int((1.0 / bbox_picked_freq) * 3)

        # Branch 1
        # If the picked video is of high quality to be used as training data, simply pick the most reliable frame
        if video_quality_score >= video_tolerance_threshold and corner_bbox_freq < 0.25:
            frames = list(track_info.keys())
            if 'meta' in frames:
                frames.remove('meta')
            video_len = len(frames)
            picked_frame_candidates_s = np.random.choice(video_len - self.memory_num, frame_candidate_num, replace=True)

            # Note that long-term frame quality is slightly different from short-term frame quality
            # For cycle memory, we also give priority to template frames with higher (T_u - T_l) frame intervals
            long_term_frame_quality_s = np.array(
                [self._calc_long_term_frame_quality(track_info[frames[cand]], video_len)
                 for cand in picked_frame_candidates_s])
            max_cand_index_s = np.argmax(long_term_frame_quality_s)
            max_cand_frame_s = int(picked_frame_candidates_s[max_cand_index_s])
            frame_id_s = frames[max_cand_frame_s]
            frame_id_s_format = '0' * (8 - len(frame_id_s)) + frame_id_s
            image_path_s_color = join(self.root, video, 'visible', "{}.{}.x.jpg".format(frame_id_s, track_id))
            image_path_s_ir = join(self.root, video, 'infrared', "{}.{}.x.jpg".format(frame_id_s, track_id))

            # Now begin to sample memory frames from nearby frames of the template frame
            # search_range = np.arange(track_info[frame_id_s][6], track_info[frame_id_s][7] + 1)
            # # First sample (memory_num + far_sample) frames in video fragment determined by DP
            # picked_frame_nearby_s = np.random.choice(search_range, self.memory_num + self.far_sample, replace=True)
            # interval_abs = np.abs(picked_frame_nearby_s - max_cand_frame_s)
            # # Pick memory_num frames "most far from" the template frame (somewhat like hard negative mining?)
            # select_idx = interval_abs.argsort()[::-1][0:self.memory_num]
            # picked_frame_nearby_s = picked_frame_nearby_s[select_idx]

            # Uncomment here to do statistics for averaged frame interval
            # print(len(search_range)-1)
            picked_frame_nearby_s = frames[max_cand_frame_s + 1 : max_cand_frame_s + self.memory_num +1]

            # frame_id_nearby_s = [frames[int(cand)] for cand in picked_frame_nearby_s]
            frame_id_nearby_s_format = ['0' * (8 - len(frame_id)) + frame_id for frame_id in picked_frame_nearby_s]
            image_path_nearby_s_color = [join(self.root, video, 'visible', "{}.{}.x.jpg".format(frame_id, track_id))
                                          for frame_id in picked_frame_nearby_s]
            image_path_nearby_s_ir = [join(self.root, video, 'infrared', "{}.{}.x.jpg".format(frame_id, track_id))
                                          for frame_id in picked_frame_nearby_s]
            bbox_nearby_s = [track_info[frame_id][:4] for frame_id in picked_frame_nearby_s]

            # Return template frame and memory frames
            return image_path_s_color, image_path_s_ir, track_info[frame_id_s][:4], image_path_nearby_s_color, image_path_nearby_s_ir, bbox_nearby_s
            #return image_path_s_color, track_info[frame_id_s][:4], image_path_nearby_s, bbox_nearby_s

        # Branch 2
        # If the picked video is not of high quality, sample video from its nearby videos
        # Step 2: re-sample video
        video_total_num = len(self.labels)
        candidate_range = np.arange(max(0, video_index - 30), min(video_total_num - 1, video_index + 31))

        # Sample from nearby videos, and pick the video with the highest quality score
        max_pick_times = 20
        video_candidate_num = 3
        max_quality_video = None
        track_id = None
        while max_pick_times:
            picked_candidates = np.random.choice(candidate_range, video_candidate_num, replace=True)
            picked_candidates_video_name = [self.videos[cand] for cand in picked_candidates]
            picked_track_id = [random.choice(list(self.labels[video_name].keys()))
                               for video_name in picked_candidates_video_name]
            video_quality_scores = np.array([self._calc_video_quality(
                self.labels[picked_candidates_video_name[cand_ind]][picked_track_id[cand_ind]]['meta'][
                    'bbox_picked_freq'],
                self.labels[picked_candidates_video_name[cand_ind]][picked_track_id[cand_ind]]['meta'][
                    'corner_bbox_freq'])
                for cand_ind in range(len(picked_candidates_video_name))])
            max_quality_index = np.argmax(video_quality_scores)
            max_quality_video = picked_candidates[max_quality_index]
            track_id = picked_track_id[max_quality_index]

            # Check if the currently selected video is of high quality or not
            if video_quality_scores[max_quality_index] > video_tolerance_threshold:
                break
            else:
                max_pick_times -= 1

        # Extreme case: if no video is picked even after 20 trials, then randomly pick one.
        if max_quality_video is None or track_id is None:
            max_quality_video = np.random.choice(candidate_range, 1)
            track_id = random.choice(list(self.labels[self.videos[max_quality_video]].keys()))

        # Re-sampling video finished
        video = self.videos[max_quality_video]
        video_info = self.labels[video]
        track_info = video_info[track_id]
        bbox_picked_freq = track_info['meta']['bbox_picked_freq']
        frame_candidate_num = int((1.0 / bbox_picked_freq) * 3)

        # Step 3: re-sample frames
        frames = list(track_info.keys())
        if 'meta' in frames:
            frames.remove('meta')
        video_len = len(frames)
        picked_frame_candidates_s = np.random.choice(video_len - self.memory_num, frame_candidate_num, replace=True)

        # Note that long-term frame quality is slightly different from short-term frame quality
        # For cycle memory, we also give priority to template frames with higher (T_u - T_l) frame intervals
        long_term_frame_quality_s = np.array([self._calc_long_term_frame_quality(track_info[frames[cand]], video_len)
                                              for cand in picked_frame_candidates_s])
        max_cand_index_s = np.argmax(long_term_frame_quality_s)
        max_cand_frame_s = picked_frame_candidates_s[max_cand_index_s]

        frame_id_s = frames[int(max_cand_frame_s)]
        frame_id_s_format = '0' * (8 - len(frame_id_s)) + frame_id_s
        image_path_s_color = join(self.root, video, 'visible', "{}.{}.x.jpg".format(frame_id_s, track_id))
        image_path_s_ir = join(self.root, video, 'infrared', "{}.{}.x.jpg".format(frame_id_s, track_id))

        # Now begin to sample memory frames from nearby frames of the template frame
        # search_range = np.arange(track_info[frame_id_s][6], track_info[frame_id_s][7] + 1)
        # # First sample (memory_num + far_sample) frames in video fragment determined by DP
        # picked_frame_nearby_s = np.random.choice(search_range, self.memory_num + self.far_sample, replace=True)
        #
        # interval_abs = np.abs(picked_frame_nearby_s - max_cand_frame_s)
        # # Pick memory_num frames "most far from" the template frame (somewhat like hard negative mining?)
        # select_idx = interval_abs.argsort()[::-1][0:self.memory_num]
        # picked_frame_nearby_s = picked_frame_nearby_s[select_idx]

        # Uncomment here to do statistics for frame interval
        # print(len(search_range)-1)
        picked_frame_nearby_s = frames[max_cand_frame_s + 1: max_cand_frame_s + self.memory_num + 1]

        # frame_id_nearby_s = [frames[int(cand)] for cand in picked_frame_nearby_s]
        frame_id_nearby_s_format = ['0' * (8 - len(frame_id)) + frame_id for frame_id in picked_frame_nearby_s]
        image_path_nearby_s_color = [join(self.root, video, 'visible', "{}.{}.x.jpg".format(frame_id, track_id))
                                      for frame_id in picked_frame_nearby_s]
        image_path_nearby_s_ir = [join(self.root, video, 'infrared', "{}.{}.x.jpg".format(frame_id, track_id))
                                      for frame_id in picked_frame_nearby_s]
        bbox_nearby_s = [track_info[frame_id][:4] for frame_id in picked_frame_nearby_s]

        # Return the template frame and memory frames
        return image_path_s_color, image_path_s_ir, track_info[frame_id_s][:4], image_path_nearby_s_color, image_path_nearby_s_ir, bbox_nearby_s

    def _get_cycle_memory_image_anno_groundTruth(self, video, track_id, video_index=None):

        """
        Loader logic for cycle memory training
        Sampling the template frame (with pseudo annotation) as well as N_mem memory frames
        """
        video_info = self.labels[video]
        track_info = video_info[track_id]

        # Sample more candidate frames if the video quality is lower, and vice versa
        frame_candidate_num = 3

        frames = list(track_info.keys())
        if 'meta' in frames:
            frames.remove('meta')
        video_len = len(frames)
        if video_len - self.memory_num < 3:
            print('video_len:', video_len)
            print('video:', video)
        picked_frame_candidates_s = np.random.choice(video_len - self.memory_num, frame_candidate_num, replace=True)

        # Note that long-term frame quality is slightly different from short-term frame quality
        # For cycle memory, we also give priority to template frames with higher (T_u - T_l) frame intervals
        max_cand_frame_s = np.random.choice(picked_frame_candidates_s)
        frame_id_s = frames[max_cand_frame_s]
        frame_id_s_format = '0' * (8 - len(frame_id_s)) + frame_id_s
        image_path_s_color = join(self.root, video, 'visible', "{}.{}.x.jpg".format(frame_id_s, track_id))
        image_path_s_ir = join(self.root, video, 'infrared', "{}.{}.x.jpg".format(frame_id_s, track_id))

        # Now begin to sample memory frames from nearby frames of the template frame

        # Uncomment here to do statistics for averaged frame interval
        # print(len(search_range)-1)
        picked_frame_nearby_s = frames[max_cand_frame_s + 1: max_cand_frame_s + self.memory_num + 1]

        # frame_id_nearby_s = [frames[int(cand)] for cand in picked_frame_nearby_s]
        frame_id_nearby_s_format = ['0' * (8 - len(frame_id)) + frame_id for frame_id in picked_frame_nearby_s]
        image_path_nearby_s_color = [join(self.root, video, 'visible', "{}.{}.x.jpg".format(frame_id, track_id))
                                     for frame_id in picked_frame_nearby_s]
        image_path_nearby_s_ir = [join(self.root, video, 'infrared', "{}.{}.x.jpg".format(frame_id, track_id))
                                  for frame_id in picked_frame_nearby_s]
        bbox_nearby_s = [track_info[frame_id][:4] for frame_id in picked_frame_nearby_s]

        # Return template frame and memory frames
        return image_path_s_color, image_path_s_ir, track_info[frame_id_s][:4], image_path_nearby_s_color, image_path_nearby_s_ir, bbox_nearby_s



def lmdb_patchFT_build_fn(cfg, lmdb, json):
    train_dataset = LMDBPatchFastTracking(cfg, lmdb_path=lmdb, json_path=json,
                                  dataset_name_list=cfg.datasets_train, num_samples=cfg.num_samples_train)
    if len(cfg.datasets_val) > 0:
        val_dataset = LMDBPatchFastTracking(cfg, lmdb_path=lmdb, json_path=json,
                                    dataset_name_list=cfg.datasets_val, num_samples=cfg.num_samples_val)
    else:
        val_dataset = None

    return train_dataset, val_dataset


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    from config import cfg_translation_track as cfg
    from register import path_register

    cfg.data.datasets_train = ['LasHeR_train', 'LasHeR_val']
    cfg.data.datasets_val = []

    trainset, valset = lmdb_patchFT_build_fn(cfg.data, lmdb=path_register.lmdb, json=path_register.json)

    train_loader = DataLoader(
        trainset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        sampler=None,
        drop_last=True,
        collate_fn=lmdb_patchFT_collate_fn
    )


    for i, image in enumerate(train_loader):
        print(i)
