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
from .scannet import ScannetDataset


class TUMRGBDStereoDataset(ScannetDataset):
    def __init__(self,
                 data_root,
                 data_split_file,
                 cleaned_seg_dir,
                 anchor_normal_dir,
                 semantic_mapping_dir,
                 pose_dir,
                 split,
                 use_new_mapping=False,
                 mode='mask',
                 transforms=None,
                 min_area=1,
                 min_h=1,
                 min_w=1,
                 mini=None):
        super(TUMRGBDStereoDataset, self).__init__(data_root,
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

        self.img_ids, self.depth_ids = self.build_list()
        self.pose_dir = pose_dir

        self.n_sample = cfg.MODEL.STEREO.N_HYPOS_PER_AXIS
        self.axis_range = cfg.MODEL.STEREO.AXIS_RANGE
        self.denser_z = cfg.MODEL.STEREO.DENSER_Z

        self.use_anchor_hypos = False

        self.pseudo_gt_dir = '/mnt/Data/jiachenliu/scannet_data/tumrgbd_pseudo_gt'

    def build_list(self):
        data_paths, depth_paths = [], []
        # save as 'scene_id/img_id' format
        with open(self.data_split_file, 'r') as fp:
            for line in fp:
                ref_path, ref_depth_path, src_path, src_depth_path = line.strip().split('\t')

                data_paths.append([ref_path, src_path])
                depth_paths.append([ref_depth_path, src_depth_path])

        return data_paths, depth_paths

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

    def build_tgt_grids(self, intrinsic, h, w, to_numpy=False):
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

        if isinstance(intrinsic, np.ndarray):
            intrinsic = torch.from_numpy(intrinsic)

        grids = torch.ones(3, h, w)
        grids[0, ...] = xs
        grids[1, ...] = ys

        camera_grid = intrinsic[:3, :3].inverse() @ grids.contiguous().view(3, -1).type_as(intrinsic)
        camera_grid = camera_grid.view(3, h, w)

        if to_numpy:
            camera_grid = camera_grid.cpu().numpy()

        return camera_grid

    def load_pseudo_gt(self, img_path, to_tensor=True, area_thresh=1000):
        scene_id = img_path.split('/')[0]
        img_id = img_path.split('/')[-1]
        save_id = scene_id + '_' + img_id.replace('png', 'npy')

        mask_path = osp.join(self.pseudo_gt_dir, 'masks', save_id)
        label_path = osp.join(self.pseudo_gt_dir, 'labels', save_id)

        if not osp.exists(mask_path):
            return None, None, None

        masks = np.load(mask_path)
        labels = np.load(label_path)

        valid_idxs = masks.sum(-1).sum(-1) > area_thresh

        if valid_idxs.sum() == 0:
            return None, None, None

        valid_masks = masks[valid_idxs]
        valid_labels = labels[valid_idxs]

        valid_bboxes = []
        for mask in valid_masks:
            bbox = self.mask_to_bbox(mask)
            valid_bboxes.append(bbox)

        valid_bboxes = np.asarray(valid_bboxes)

        if to_tensor:
            valid_masks = torch.from_numpy(valid_masks)
            valid_bboxes = torch.from_numpy(valid_bboxes)
            valid_labels = torch.from_numpy(valid_labels)

        return valid_masks, valid_bboxes, valid_labels

    def __len__(self):
        return len(self.img_ids)

    def load_intrinsic(self, h, w, ori_h, ori_w):
        intrinsic = np.asarray([
            [525.0, 0, 319.5],
            [0, 525.0, 239.5],
            [0, 0, 1]
        ]).astype(np.float)

        intrinsic[0][0] = intrinsic[0][0] * w / ori_w
        intrinsic[1][1] = intrinsic[1][1] * h / ori_h

        intrinsic[0][2] = w / 2
        intrinsic[1][2] = h / 2

        return intrinsic

    def __getitem__(self, idx, visualize=False):
        ref_path, src_path = self.img_ids[idx]

        scene_id = ref_path.split('/')[0]
        ref_img_id = ref_path.split('/')[-1]

        src_scene_id = src_path.split('/')[0]
        src_img_id = src_path.split('/')[-1]

        h = self.cfg.INPUT.MIN_SIZE_TRAIN[0]
        w = int(h * 4 / 3)

        ref_img_path = osp.join(self.data_root, scene_id, 'rgb', ref_img_id)
        if not osp.exists(ref_img_path):
            images, target, _ = self[(idx + 1) % len(self)]

            return images, target, idx

        src_img_path = osp.join(self.data_root, scene_id, 'rgb', src_img_id)
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

        target = {}

        ref_depth_id = self.depth_ids[idx][0].split('/')[-1]
        ref_depth_path = osp.join(self.data_root, scene_id, 'depth', ref_depth_id)

        if not osp.exists(ref_depth_path):
            images, target, _ = self[(idx + 1) % len(self)]
            return images, target, idx

        depth = self.load_depth(ref_depth_path, divisor=5000.0)

        masks, bboxes, labels = self.load_pseudo_gt(ref_path, to_tensor=True)

        if masks is None:
            images, target, _ = self[(idx + 1) % len(self)]

            return images, target, idx

        target = BoxList(bboxes, ref_img.size, mode='xyxy')
        masks = SegmentationMask(masks, ref_img.size, mode=self.mode)

        target.add_field('masks', masks)
        target.add_field('labels', labels.long())

        target.add_field('depth', depth)

        anchor_normals = np.load(self.anchor_normal_dir)
        target.add_field('anchor_normals', torch.from_numpy(anchor_normals))

        target.add_field('sem_mapping', self.id_to_label)

        stereo_h = self.cfg.MODEL.STEREO.STEREO_H
        stereo_w = self.cfg.MODEL.STEREO.STEREO_W

        if self.cfg.MODEL.STEREO.POOL_FEATURE:
            stereo_h = stereo_h // 2
            stereo_w = stereo_w // 2

        # intrinsic with image size
        # if not self.is_train:
        intrinsic = self.load_intrinsic(h, w, ori_h, ori_w)
        target.add_field('intrinsic', torch.from_numpy(intrinsic))

        img_tgt_camera_grids = self.build_tgt_grids(intrinsic, h, w)
        target.add_field('img_camera_grid', img_tgt_camera_grids)

        ref_pose_path = osp.join(self.pose_dir, scene_id, ref_img_id.replace('png', 'txt'))
        ref_pose = self.load_pose(ref_pose_path)

        if not self.is_train:
            target.add_field('ref_pose', torch.from_numpy(ref_pose))

        src_pose_path = osp.join(self.pose_dir, scene_id, src_img_id.replace('png', 'txt'))
        src_pose = self.load_pose(src_pose_path)

        intrinsic_for_stereo = self.load_intrinsic(stereo_h, stereo_w, ori_h, ori_w)
        homo_grid, hypos = self.build_homo_grids(ref_pose, src_pose, intrinsic_for_stereo)
        images['homo_grid'] = torch.from_numpy(homo_grid)
        images['hypos'] = torch.from_numpy(hypos)

        if self.transforms is not None:
            processed_img, target = self.transforms(ref_img, target)
            images['ref_img'] = processed_img

            processed_img, _ = self.transforms(src_img, target)
            images['src_img'] = processed_img

        return images, target, idx
