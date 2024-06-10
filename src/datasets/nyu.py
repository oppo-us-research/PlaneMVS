"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import os.path as osp

import cv2

import numpy as np
from PIL import Image

import torch

from src.config import cfg


class NYUDepthDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 data_split_file,
                 anchor_normal_dir,
                 split):
        assert split in ['train', 'val', 'test']

        self.cfg = cfg
        self.data_split_file = data_split_file
        self.color_dir = osp.join(data_root, 'color_rgb')
        self.depth_dir = osp.join(data_root, 'depth_npy')

        self.is_train = split == 'train'

        self.anchor_normal_dir = anchor_normal_dir

        self.img_ids = self.build_list()

    def build_list(self):
        data_paths = []

        with open(self.data_split_file, 'r') as fp:
            for line in fp:
                data_paths.append(line.strip())

        return data_paths

    def load_depth(self, depth_path, divisor=1000.0):
        if depth_path.endswith('.npy'):
            depth_map = np.load(depth_path)

        elif depth_path.endswith('.png'):
            depth_map = cv2.imread(depth_path, -1) / divisor

        else:
            raise NotImplementedError

        return depth_map

    def load_intrinsic(self, h=480, w=640):
        intrinsic = np.array([
            [528.81872559, 0, 320, 0],
            [0, 529.44720459, 240, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        intrinsic[0][0] = intrinsic[0][0] * w / 640
        intrinsic[1][1] = intrinsic[1][1] * w / 480

        return intrinsic

    def load_image(self, image_path, h=480, w=640):
        img = Image.open(image_path)
        img = img.resize((w, h)).convert('RGB')

        return img

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img_path = osp.join(self.color_dir, img_id)
        img = self.load_image(img_path)

        depth_path = osp.join(self.depth_dir, img_id)
        depth = self.load_depth(depth_path)

        print(img.shape, depth.shape)
        exit(1)

        images = {}
        if not self.is_train:
            images['ori_img'] = np.asarray(img)

        else:
            images['ori_img'] = img

        images['path'] = img_path

        target = {}
        target['depth'] = depth

        if self.transforms is not None:
            processed_img, target = self.transform(img, target)

            images['img'] = processed_img

        return images, target
