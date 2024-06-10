"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import os.path as osp
import json
import cv2

import numpy as np
from scipy import stats

from PIL import Image

import torch
import torch.nn.functional as F


""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


""" load our own modules """
from src.config import cfg
from src.utils.colormap import ColorPalette
from .scannet import ScannetDataset


class ScannetStereoInferenceDataset(ScannetDataset):
    def __init__(self,
                 data_root,
                 data_split_file,
                 cleaned_seg_dir,
                 anchor_normal_dir,
                 semantic_mapping_dir,
                 pixel_plane_dir,
                 split,
                 use_new_mapping=False,
                 mode='mask',
                 transforms=None,
                 min_area=1,
                 min_h=1,
                 min_w=1,
                 mini=None):
        super(ScannetStereoInferenceDataset, self).__init__(data_root,
                                                            data_split_file,
                                                            cleaned_seg_dir,
                                                            anchor_normal_dir,
                                                            semantic_mapping_dir,
                                                            split,
                                                            use_new_mapping,
                                                            mode,
                                                            transforms,
                                                            min_area,
                                                            min_h,
                                                            min_w,
                                                            mini)
        assert split in ['val', 'test']

        # plane hypos num sampled per axis
        self.n_sample = cfg.MODEL.STEREO.N_HYPOS_PER_AXIS

        # the starting and ending range of the axis
        self.axis_range = cfg.MODEL.STEREO.AXIS_RANGE

        # whether to use denser z axis since z distributes mainly from (-2, 0.5)
        self.denser_z = cfg.MODEL.STEREO.DENSER_Z

        self.img_ids = self.build_list()

    def build_list(self):
        data_paths = []
        # save as 'scene_id/img_id' format
        with open(self.data_split_file, 'r') as fp:
            for line in fp:
                ref_path, src_path = line.strip().split('\t')
                data_paths.append([ref_path, src_path])

        return data_paths

    def load_plane(self, plane_path, plane_idxs):
        all_planes = np.load(plane_path)

        planes = all_planes[plane_idxs]

        return planes

    # load pixel plane fitted by gt depth
    def load_pixel_plane(self, pixel_plane_path, pixel_plane_mask_path, h, w):
        pixel_plane = np.load(pixel_plane_path)
        pixel_plane_mask = np.load(pixel_plane_mask_path)

        pixel_plane = cv2.resize(pixel_plane, (w, h), cv2.INTER_LINEAR)

        pixel_plane_mask = cv2.resize(pixel_plane_mask.astype(np.float64), (w, h), cv2.INTER_LINEAR)
        pixel_plane_mask = pixel_plane_mask > 0.5

        return pixel_plane, pixel_plane_mask

    # transform into n/d plane map
    def transform_normal_div_offset(self, planes, pose, masks):
        ds = np.linalg.norm(planes, axis=-1, keepdims=True)
        ns = planes / (ds + 1e-10)

        plane_instances = []

        world_planes = np.concatenate([ns, -ds], axis=-1)
        camera_planes = world_planes @ pose

        # here d means nx - d = 0, it should be a bug
        # n_div_ds = camera_planes[:, :3] / (-camera_planes[:, -1:] + 1e-10)

        # here d means nx + d = 0, then the homo should be correct
        n_div_ds = camera_planes[:, :3] / (camera_planes[:, -1:] + 1e-10)

        n_div_d_map = np.zeros((*masks[0].shape, 3))

        for n_div_d, mask in zip(n_div_ds, masks):
            n_div_d_map[mask, :] = n_div_d

        plane_instances = n_div_ds

        return n_div_d_map, plane_instances

    # plane homography computation
    def build_homo_grids(self, pose_tgt, pose_src, intrinsic):
        ranges = np.array(np.linspace(-self.axis_range, self.axis_range, self.n_sample))
        ranges_x = np.tile(ranges.reshape(self.n_sample, 1, 1), (1, self.n_sample, self.n_sample))
        ranges_y = np.tile(ranges.reshape(1, self.n_sample, 1), (self.n_sample, 1, self.n_sample))

        if self.denser_z:
            ranges_z = np.array(np.linspace(-2, 0.5, self.n_sample))
            ranges_z = np.tile(ranges_z.reshape(1, 1, self.n_sample), (self.n_sample, self.n_sample, 1))

        else:
            ranges_z = np.tile(ranges.reshape(1, 1, self.n_sample), (self.n_sample, self.n_sample, 1))

        # (n_sample, n_sample, n_sample, 3)
        hypos_vol = np.stack([ranges_x, ranges_y, ranges_z], axis=-1)

        rot_tgt = pose_tgt[:3, :3]
        trans_tgt = pose_tgt[:3, -1:]

        rot_src = pose_src[:3, :3]
        trans_src = pose_src[:3, -1:]

        # (3, 3) @ (3, 3) + (3, 3) @ (3, 1) @ (n, 1, 3)
        homo = rot_src.T @ rot_tgt + (rot_src.T @ (trans_src - trans_tgt)) @ hypos_vol.reshape(-1, 3)[:, None, :]
        homo = intrinsic[:3, :3] @ homo @ np.linalg.inv(intrinsic[:3, :3])

        hypos_vol = hypos_vol.reshape(-1, 3)

        return homo, hypos_vol

    # for visualization and debug
    def build_tgt_grids(self, intrinsic, h, w, to_numpy=False):
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

        if isinstance(intrinsic, np.ndarray):
            intrinsic = torch.from_numpy(intrinsic)

        grids = torch.ones(3, h, w)
        grids[0, ...] = xs
        grids[1, ...] = ys

        # [3, h, w]
        camera_grid = intrinsic[:3, :3].inverse() @ grids.contiguous().view(3, -1).type_as(intrinsic)
        camera_grid = camera_grid.view(3, h, w)
        if to_numpy:
            camera_grid = camera_grid.cpu().numpy()

        return camera_grid

    def warp_src_to_ref(self, depth, intrinsic, ref_pose, src_pose, src_img):
        h, w = depth.size()[-2:]

        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

        grids = torch.ones(3, h, w)
        grids[0, ...] = xs
        grids[1, ...] = ys

        if isinstance(intrinsic, np.ndarray):
            intrinsic = torch.from_numpy(intrinsic).type_as(grids)

        if isinstance(ref_pose, np.ndarray):
            ref_pose = torch.from_numpy(ref_pose).type_as(grids)
            src_pose = torch.from_numpy(src_pose).type_as(grids)

        camera_grid = intrinsic[:3, :3].inverse() @ grids.contiguous().view(3, -1).type_as(intrinsic)
        camera_grid = camera_grid.view(3, h, w)

        points = camera_grid * depth

        homo_ref_points = torch.ones(4, h, w)
        homo_ref_points[:3, ...] = points

        homo_src_points = src_pose.inverse() @ ref_pose @ homo_ref_points.view(4, -1)

        src_points = (intrinsic[:3, :3] @ homo_src_points[:3, :]).view(3, h, w)

        src_xys = src_points[:2, ...] / src_points[-1:, ...]
        src_xys = src_xys.permute(1, 2, 0).unsqueeze(dim=0)

        workspace = torch.tensor([(w-1) / 2, (h-1) / 2]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        normalized_src_xys = src_xys / workspace - 1

        src_img = torch.from_numpy(np.asarray(src_img)).permute(2, 0, 1).unsqueeze(dim=0).float()

        warped_src = F.grid_sample(
            src_img, normalized_src_xys.type_as(src_img), mode='bilinear', 
            padding_mode='zeros', align_corners=True)
        warped_src = warped_src.squeeze(dim=0).permute(1, 2, 0).cpu().numpy()

        return warped_src

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx, visualize=False):
        ref_path, src_path = self.img_ids[idx]

        scene_id, ref_img_id = ref_path.split('/')
        _, src_img_id = src_path.split('/')

        h = self.cfg.INPUT.MIN_SIZE_TRAIN[0]
        w = int(h * 4 / 3)

        ref_img_path = osp.join(self.data_root, scene_id, self.color_dir, ref_img_id)
        if not osp.exists(ref_img_path):
            images, target, _ = self[(idx + 1) % len(self)]

            return images, target, idx

        src_img_path = osp.join(self.data_root, scene_id, self.color_dir, src_img_id)
        if not osp.exists(src_img_path):
            images, target, _ = self[(idx + 1) % len(self)]

        ref_img = Image.open(ref_img_path)
        ori_w, ori_h = ref_img.size
        ref_img = ref_img.resize((w, h)).convert('RGB')

        src_img = Image.open(src_img_path).resize((w, h)).convert('RGB')

        images = {}
        if not self.is_train:
            images['ori_ref_img'] = np.asarray(ref_img)
            images['ori_src_img'] = np.asarray(src_img)
        else:
            images['ori_ref_img'] = ref_img
            images['ori_src_img'] = src_img

        images['ref_path'] = ref_path
        images['src_path'] = src_path

        target = BoxList(torch.zeros(1, 4), ref_img.size, mode='xyxy')

        ref_depth_path = osp.join(self.data_root, scene_id, self.depth_dir, ref_img_id.replace('jpg', 'png'))
        depth = self.load_depth(ref_depth_path)

        target.add_field('depth', depth)

        if not self.is_train:
            target.add_field('sem_mapping', self.id_to_label)

        # feature map size used as stereo input
        stereo_h = self.cfg.MODEL.STEREO.STEREO_H
        stereo_w = self.cfg.MODEL.STEREO.STEREO_W

        # if we do pooling for feature map, we need to further rescale the intrinsic
        if self.cfg.MODEL.STEREO.POOL_FEATURE:
            stereo_h = stereo_h // 2
            stereo_w = stereo_w // 2

        intrinsic_path = osp.join(self.data_root, scene_id, self.intrinsic_dir, 'intrinsic_color.txt')
        # intrinsic with feature map size
        intrinsic_for_stereo = self.load_intrinsic(intrinsic_path, stereo_h, stereo_w, ori_h, ori_w)
        target.add_field('intrinsic_for_stereo', torch.from_numpy(intrinsic_for_stereo))

        # intrinsic with image size
        intrinsic = self.load_intrinsic(intrinsic_path, h, w, ori_h, ori_w)
        target.add_field('intrinsic', torch.from_numpy(intrinsic))

        ref_pose_path = osp.join(self.data_root, scene_id, self.pose_dir, ref_img_id.replace('jpg', 'txt'))
        ref_pose = self.load_pose(ref_pose_path)
        target.add_field('ref_pose', torch.from_numpy(ref_pose))

        src_pose_path = osp.join(self.data_root, scene_id, self.pose_dir, src_img_id.replace('jpg', 'txt'))
        src_pose = self.load_pose(src_pose_path)
        target.add_field('src_pose', torch.from_numpy(src_pose))

        warped_src = self.warp_src_to_ref(depth, intrinsic, ref_pose, src_pose, src_img)
        images['warped_src_img'] = warped_src.astype(np.uint8)

        # plane hypos and homography matrixs
        homo_grid, hypos = self.build_homo_grids(ref_pose, src_pose, intrinsic_for_stereo)
        images['homo_grid'] = torch.from_numpy(homo_grid)
        images['hypos'] = torch.from_numpy(hypos)

        if self.transforms is not None:
            processed_img, target = self.transforms(ref_img, target)
            images['ref_img'] = processed_img

            processed_img, _ = self.transforms(src_img, target)
            images['src_img'] = processed_img

        return images, target, idx
