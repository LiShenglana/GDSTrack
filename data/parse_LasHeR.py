import argparse
import os
import sys

import numpy as np

sys.path.append('../')
from lib.dataset import LMDBData, SubSet, DatasetLoader, parse_loop

"""
Examples:

GOT-10k/
├── test
├── train
├── val
└── vot19-20_got10k_prohibited_1000.txt


frame_dict_list = [
    {'bbox': [409.0, 428.0, 1509.0, 312.0], 
    'dataset': 'got10k_train-vot', 
    'key': 'GOT-10k_Train_001281/00000007', 
    'length': 100, 
    'path': 'GOT-10k_Train_001281/00000007.jpg', 
    'size': [1920, 1080], 
    'video': 'GOT-10k_Train_001281'}, 
     ...
]
"""


class GOT10kLoader(DatasetLoader):
    def __init__(self, name: str, split: str):
        super(GOT10kLoader, self).__init__()

        self.dataset_name = name

        # if 'train' in split:
        #     self.root = '/media/cscv/d00985a0-c3e6-4ffa-9546-88c861db5ce3/02_Dataset/LasHeR/trainingset_ori'
        # elif 'val' in split:
        #     self.root = '/media/cscv/d00985a0-c3e6-4ffa-9546-88c861db5ce3/02_Dataset/LasHeR/valset'
        # else:
        #     self.root = '/media/cscv/d00985a0-c3e6-4ffa-9546-88c861db5ce3/02_Dataset/LasHeR/testingset'
            
        self.root = '/media/cscv/d00985a0-c3e6-4ffa-9546-88c861db5ce3/02_Dataset/LasHeR/trainingset_ori'
        self.data_dir = self.root

        if split == 'test' or split == 'train' or split == 'val':
            with open(os.path.join(self.root, 'list.txt'), 'r') as f:
                video_list = f.readlines()
            self.video_list = sorted([x.split('\n')[0] for x in video_list])
        else:
            raise NotImplementedError

        if 'vot' in split:
            with open('vot_got10k_prohibited_1000.txt', 'r') as f:
                vot_exc = f.readlines()
            vot_exc = sorted([x.split('\n')[0] for x in vot_exc])
            self.video_list = [x for x in self.video_list if x not in vot_exc]

    def get_video_info(self, video_name: str):
        self.video_name = video_name
        video_dir = os.path.join(self.data_dir, video_name)
        color_video_dir = os.path.join(video_dir, 'visible')
        ir_video_dir = os.path.join(video_dir, 'infrared')

        gt_list = np.loadtxt(os.path.join(video_dir, 'init.txt'), delimiter=',')
        self.gt_list = gt_list.reshape(-1, 4)

        color_img_list = sorted(os.listdir(color_video_dir))
        color_img_list = np.array([os.path.join(video_name, 'visible', img_file) for img_file in color_img_list if '.jpg' in img_file])
        self.color_img_list = color_img_list[:gt_list.shape[0]]  # used as the key of lmdb
        self.color_key_list = np.array([f.split('.')[0] for f in self.color_img_list])

        ir_img_list = sorted(os.listdir(ir_video_dir))
        ir_img_list = np.array(
            [os.path.join(video_name, 'infrared', img_file) for img_file in ir_img_list if '.jpg' in img_file])
        self.ir_img_list = ir_img_list[:gt_list.shape[0]]  # used as the key of lmdb
        self.ir_key_list = np.array([f.split('.')[0] for f in self.ir_img_list])

        self.lang_list = None

        # absent = np.loadtxt(os.path.join(video_dir, 'absence.label'))
        # self.gt_list = self.gt_list[absent == 0]
        # self.img_list = self.img_list[absent == 0]

        self.imw, self.imh = None, None


if __name__ == '__main__':
    # #################################
    # [got10k_train] -- 9335 videos, 1401877 frames, done !
    # [got10k_train_vot] -- 8335 videos, 1250652 frames, done !
    # [got10k_val] -- 180 videos, 20979 frames, done !
    # #################################
    parser = argparse.ArgumentParser(description='parse got10k for lmdb')
    parser.add_argument('--split', default='train', type=str, choices=['train', 'test', 'val'],
                        help='select training set or testing set')
    parser.add_argument('--dir', default='/media/cscv/d00985a0-c3e6-4ffa-9546-88c861db5ce3/02_Dataset/LasHeR/LMDB', type=str)
    parser.add_argument('--only_json', dest='only_json', action='store_true', default=False)
    args = parser.parse_args()

    dataset_name = f'LasHeR_{args.split}'
    save_dir = os.path.join(args.dir, dataset_name)

    if args.only_json:
        lmdb_dict = None
        if os.path.exists(os.path.join(save_dir, f'{dataset_name}.json')):
            print(f'Directory not empty: {dataset_name}.json had been built.')
            sys.exit()
        else:
            os.makedirs(save_dir, exist_ok=True)
    else:
        lmdb_dict = LMDBData(save_dir=save_dir)
    data_info = SubSet(name=dataset_name)
    data_set = GOT10kLoader(name=dataset_name, split=args.split)

    parse_loop(name=dataset_name, save_dir=save_dir,
               data_set=data_set, json_obj=data_info, lmdb_obj=lmdb_dict)
