# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

from multiprocessing import Value

from logging import getLogger

import torch
import numpy as np

_GLOBAL_SEED = 0
logger = getLogger()


class generate_bbox(object):

    def __init__(
        self,
        input_size=(128, 128),
        enc_mask_scale=(0.1, 0.3),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        min_keep=1,
        allow_overlap=False
    ):
        super(generate_bbox, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height_, self.width_ = input_size[0], input_size[1]
        self.enc_mask_scale = enc_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height_ * self.width_ * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height_:
            h -= 1
        while w >= self.width_:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size):
        h, w = b_size

        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1

            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        bbox = np.array([left.item(), top.item(), w, h], dtype=np.int64).squeeze()
        mask = mask.squeeze()

        # --
        return mask, bbox

    def generate(self, x):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        input_size = x.shape
        self.height, self.width = input_size[0], input_size[1]
        
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)

        
        mask, bbox = self._sample_block_mask(e_size)


        return bbox