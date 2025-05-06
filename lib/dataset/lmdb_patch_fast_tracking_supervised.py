import math

import cv2
import random
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import lmdb
import albumentations as aug
from copy import deepcopy
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import torch
from lib.dataset._dataset import SubSet, BaseDataset
from lib.dataset.crop_image_visualize import crop_image_visualize
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

        # load dataset
        self.LMDB_ENVS = {}
        self.LMDB_HANDLES = {}
        self.dataset_list: List = []
        for name in dataset_name_list:
            dataset = SubSet(name=name, load=self.json_path[name])
            self.dataset_list.append([dataset.data_set, len(dataset.data_set)])

            env = lmdb.open(self.lmdb_path[name], readonly=True, lock=False, readahead=False, meminit=False)
            self.LMDB_ENVS[name] = env
            item = env.begin(write=False)
            # for key, value in item.cursor():
            #     print(key, value)
            self.LMDB_HANDLES[name] = item

        self.num_samples = num_samples


    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):

        video_list, list_len = random.choice(self.dataset_list)
        idx = np.random.randint(0, list_len)

        t_dict, s1_dict, s2_dict = self.check_sample(video_list[idx], video_list, self.sample_range)

        # read RGB image, [x y w h]
        _template_img_color, _template_img_ir, _t_box, t_lang = self.parse_frame_lmdb(t_dict, self.LMDB_HANDLES, need_language=True)
        _search1_img_color, _search1_img_ir, _s1_box, s1_lang = self.parse_frame_lmdb(s1_dict, self.LMDB_HANDLES, need_language=True)
        _search2_img_color, _search2_img_ir, _s2_box, s2_lang = self.parse_frame_lmdb(s2_dict, self.LMDB_HANDLES,
                                                                                      need_language=True)

        # generator_bbox = generate_bbox()
        # _t_box = generator_bbox.generate(_template_img_color)
        # _s1_box = deepcopy(_t_box)
        # _s2_box = deepcopy(_t_box)
        # _s3_box = deepcopy(_t_box)
        #
        # crop_image_visualize(_template_img_color, _template_img_ir, _t_box, _search1_img_color, _search1_img_ir, _s1_box,
        #                      _search2_img_color, _search2_img_ir, _s2_box, _search2_img_color, _search2_img_ir, _s3_box)
        template_img_color, template_img_ir, _, t_box, shift_t = self.crop_patch_fast(
            _template_img_color, _template_img_ir, _t_box,
            out_size=self.template_sz, padding=0, #out_size=self.template_sz
            scale_factor=self.template_scale_f, #self.template_scale_f,
            jitter_f=self.template_jitter_f) #self.template_jitter_f

        search1_img_color, search1_img_ir, _, s1_box, shift_s1 = self.crop_patch_fast(
            _search1_img_color, _search1_img_ir, _s1_box,
            out_size=self.search_sz, padding=0,
            scale_factor=self.search_scale_f,#self.search_scale_f,
            jitter_f=self.search_jitter_f)

        search2_img_color, search2_img_ir, _, s2_box, shift_s2 = self.crop_patch_fast(
            _search2_img_color, _search2_img_ir, _s2_box,
            out_size=self.search_sz, padding=0,
            scale_factor=self.search_scale_f,#self.search_scale_f,
            jitter_f=self.search_jitter_f)


        # crop_image_visualize(search1_img_color, search1_img_ir, s1_box, search2_img_color, search2_img_ir,
        #                      s2_box, search3_img_color, search3_img_ir, s3_box, search3_img_color, search3_img_ir,
        #                      s3_box)

        # template_img_color, template_img_ir, search1_img_color, search1_img_ir, search2_img_color, search2_img_ir, search3_img_color, search3_img_ir = self._augmentation(template_img_color, template_img_ir, search1_img_color, search1_img_ir, search2_img_color, search2_img_ir, search3_img_color, search3_img_ir)
        template_img_color, template_img_ir, search1_img_color, search1_img_ir, search2_img_color, search2_img_ir = map(lambda im: self.aug(image=im)["image"], [template_img_color, template_img_ir, search1_img_color, search1_img_ir, search2_img_color, search2_img_ir])

        a = np.random.rand(2, 1)

        s1_box_ori = deepcopy(s1_box)
        s2_box_ori = deepcopy(s2_box)

        if a[0] < 0.5:
            search1_img_color, s1_box = self.horizontal_flip(search1_img_color, s1_box_ori)
            search1_img_ir = self.horizontal_flip(search1_img_ir)

        if a[1] < 0.5:
            search2_img_color, s2_box = self.horizontal_flip(search2_img_color, s2_box_ori)
            search2_img_ir = self.horizontal_flip(search2_img_ir)

        # crop_image_visualize(template_img_color, template_img_ir, t_box, search1_img_color, search1_img_ir, s1_box, search2_img_color, search2_img_ir,
        #                      s2_box, search3_img_color, search3_img_ir, s3_box)


        test_outbox = False
        if test_outbox:
            im_show1 = cv2.cvtColor(search1_img_color, cv2.COLOR_RGB2BGR)
            cv2.rectangle(im_show1, (int(s1_box[0]), int(s1_box[1])), (int(s1_box[0]) + int(s1_box[2]), int(s1_box[1]) + int(s1_box[3])), (0, 255, 0),
                          3)
            im_show2 = cv2.cvtColor(search2_img_color, cv2.COLOR_RGB2BGR)
            cv2.rectangle(im_show2, (int(s2_box[0]), int(s2_box[1])), (int(s2_box[0]) + int(s2_box[2]), int(s2_box[1]) + int(s2_box[3])), (0, 255, 0),
                          3)
            cv2.imshow('template_img_color', im_show1)
            cv2.imshow('search1_img_color', im_show2)
            cv2.waitKey()
        # crop_image_visualize(search1_img_color, search1_img_ir, s1_box, search2_img_color, search2_img_ir,
        #                          s2_box, search3_img_color, search3_img_ir, s3_box, search3_img_color, search3_img_ir,
        #                          s3_box)

        if self.debug:
            print(t_box.astype(int), s1_box.astype(int), s1_lang)
            self.debug_fn([template_img_color, search1_img_color], [t_box, s1_box])
            self.debug_fn([template_img_ir, search1_img_ir], [t_box, s1_box])


        template_img_color, search1_img_color, search2_img_color = map(lambda x: x.transpose(2, 0, 1).astype(np.float64), [template_img_color, search1_img_color, search2_img_color])
        template_img_ir, search1_img_ir, search2_img_ir = map(lambda x: x.transpose(2, 0, 1).astype(np.float64),
                                       [template_img_ir, search1_img_ir, search2_img_ir])
        t_box, s1_box, s2_box = map(lambda x: x.astype(np.float64), [t_box, s1_box, s2_box])  # [x, y, w, h]

        s1_box[2:] = s1_box[:2] + s1_box[2:] - 1
        s1_box[0::2] = s1_box[0::2] / self.search_sz[1]
        s1_box[1::2] = s1_box[1::2] / self.search_sz[0]

        s2_box[2:] = s2_box[:2] + s2_box[2:] - 1
        s2_box[0::2] = s2_box[0::2] / self.search_sz[1]
        s2_box[1::2] = s2_box[1::2] / self.search_sz[0]

        return template_img_color, template_img_ir, s1_box, search1_img_color, search1_img_ir, s2_box, search2_img_color, search2_img_ir

    def _augmentation(self, template_img_color, template_img_ir, search1_img_color, search1_img_ir, search2_img_color, search2_img_ir, search3_img_color, search3_img_ir, search=False, cycle_memory=False):
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

        search_images1 = torch.cat(
            (torch.tensor(search1_img_color).unsqueeze(0), torch.tensor(search1_img_ir).unsqueeze(0)),
            0).numpy()  # [2,128,128,3]
        search_images1 = self.search_aug_seq1(images=search_images1)

        search_images2 = torch.cat(
            (torch.tensor(search2_img_color).unsqueeze(0), torch.tensor(search2_img_ir).unsqueeze(0)),
            0).numpy()  # [2,128,128,3]
        search_images2 = self.search_aug_seq2(images=search_images2)

        search_images3 = torch.cat(
            (torch.tensor(search3_img_color).unsqueeze(0), torch.tensor(search3_img_ir).unsqueeze(0)),
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

        return template_images[0], template_images[1], search_images1[0], search_images1[1], search_images2[0], search_images2[1], search_images3[0], search_images3[1]#, bbox, param

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
