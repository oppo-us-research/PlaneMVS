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
from src.tools.utils import print_indebugmode

from typing import Tuple, Union, List, Type, Dict


class ScannetStereoDataset(ScannetDataset):
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
        
        super(ScannetStereoDataset, self).__init__(data_root,
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
        assert split in ['train', 'val', 'test']

        # plane hypos num sampled per axis
        self.n_sample = cfg.MODEL.STEREO.N_HYPOS_PER_AXIS

        # the starting and ending range of the axis;
        # this is used to sample the pixel-level plane parameters p_i = n_i^T / d_i;
        # where (n_i)^T p_i + d_i = 0, n_i is the plane normal and d_i is the offset 
        # at pixel p_i of the reference view.
        self.axis_range = cfg.MODEL.STEREO.AXIS_RANGE
        
        # whether to use denser z axis since z (actually, it is n_z/d) 
        # distributes mainly from (-2, 0.5);
        self.denser_z = cfg.MODEL.STEREO.DENSER_Z

        # pixel plane fitted by gt depth, can provide extra supervision
        self.pixel_plane_dir = pixel_plane_dir

        # whether to provide pixel plane labels
        self.train_pixel_plane = cfg.MODEL.STEREO.TRAIN_PIXEL_PLANE

        # whether to load pixel gt plane map, generated from gt depth
        self.with_pixel_gt_plane_map = cfg.MODEL.STEREO.WITH_PIXEL_GT_PLANE_MAP

        # whether to add instance-level planar depth map supervision
        self.with_instance_planar_depth_loss = self.cfg.MODEL.STEREO.INSTANCE_PLANAR_DEPTH_LOSS or \
                                               self.cfg.MODEL.STEREO.PRED_INSTANCE_PLANAR_DEPTH_LOSS

        self.use_all_data = self.cfg.MODEL.STEREO.USE_ALL_DATA
        self.split = split

        if self.use_all_data and self.split == 'train':
            self.data_split_file = 'data_splits/valid_stereo_train_files.txt'
            assert osp.exists(self.data_split_file)

        self.is_test_split = 'test' in self.data_split_file

        self.img_ids = self.build_list()

    def build_list(self):
        data_paths = []
        # save as 'scene_id/img_id' format
        with open(self.data_split_file, 'r') as fp:
            for line in fp:
                # e.g, a line:
                # scene0011_00/1378.jpg	scene0011_00/1383.jpg
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

        # here d means nx + d = 0, then the homo should be correct
        n_div_ds = camera_planes[:, :3] / (camera_planes[:, -1:] + 1e-10)

        n_div_d_map = np.zeros((*masks[0].shape, 3))

        for n_div_d, mask in zip(n_div_ds, masks):
            n_div_d_map[mask, :] = n_div_d

        plane_instances = n_div_ds

        return n_div_d_map, plane_instances
    
    
    def build_homo_grids(self, 
            pose_tgt: np.ndarray, 
            pose_src: np.ndarray, 
            intrinsic: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
        """ plane homography computation. 
            See Eq (1) in the main paper https://arxiv.org/abs/2203.12082;
        Args:
            pose_tgt: 3x3 or 4x4 matrix, target or reference camera (view) pose (camera-to-world pose, 
                      i.e., the inverse matrix of camera extrinsic);
            pose_src: 3x3 or 4x4 matrix, source camera (view) pose (camera-to-world pose, 
                      i.e., the inverse matrix of camera extrinsic);
            intrinsic: 3x3 or 4x4 camera intrinsic matrix. It usually holds true that
                        reference view and source view has the same intrinsic. 
        
        Returns:
            hypos_vol: a 2D array with size (K*K*K, 3), for slanted plane hypotheses of n^T/d;
                        here K is the number of samples among each axis (x,y,z) of n^T/d.
                        E.g., K=8, i.e., hypos_vol with shape (512, 3)
            
            homo: 3D array with size (K*K*K, 3, 3), each 3x3 homography matrix will map pixel in 
                  the target/reference view to the source view given current n_i^T / d_i;
        """


        """ slanted plane parameters n_i^T / d_i distribution """
        # this is used to sample the pixel-level plane parameters p_i = n_i^T / d_i;
        # where (n_i)^T * p_i + d_i = 0, n_i is the plane normal and d_i is the offset 
        # at pixel p_i of the reference view.
        
        # See Section: "Hypothesis selection for slanted planes" in paper Supplementary:
        # here "n" is the plane normal vector, "d" is the offset;
        # 1) we select (-2, 2) and (-2, 2) as the range of x and y axis for n^T/d;
        # 2) For the z axis of n^T/d, we select (-2, 0.5) if self.denser_z is True, 
        #     or (-2, 2) otherwise;
        #     In our experiments, we set self.denser_z= True for the range (-2, 0.5),
        #     which gives more accurate results than (-2, 2).
        ranges = np.array(np.linspace(-self.axis_range, self.axis_range, self.n_sample))
        ranges_x = np.tile(ranges.reshape(self.n_sample, 1, 1), (1, self.n_sample, self.n_sample))
        ranges_y = np.tile(ranges.reshape(1, self.n_sample, 1), (self.n_sample, 1, self.n_sample))

        if self.denser_z:
            ranges_z = np.array(np.linspace(-2, 0.5, self.n_sample))
            ranges_z = np.tile(ranges_z.reshape(1, 1, self.n_sample), (self.n_sample, self.n_sample, 1))

        else:
            ranges_z = np.tile(ranges.reshape(1, 1, self.n_sample), (self.n_sample, self.n_sample, 1))

        # (K, K, K, 3) --> (K*K*K, 3), where K=self.n_sample (e.g., =8);
        hypos_vol = np.stack([ranges_x, ranges_y, ranges_z], axis=-1).reshape(-1, 3)

        rot_tgt = pose_tgt[:3, :3]
        trans_tgt = pose_tgt[:3, -1:]

        rot_src = pose_src[:3, :3]
        trans_src = pose_src[:3, -1:]

        # (3, 3) @ (3, 3) + (3, 3) @ (3, 1) @ (n, 1, 3)
        homo = rot_src.T @ rot_tgt + (rot_src.T @ (trans_src - trans_tgt)) @ hypos_vol[:, None, :]
        # [K*K*K, 3, 3]
        homo = intrinsic[:3, :3] @ homo @ np.linalg.inv(intrinsic[:3, :3]) 

        return homo, hypos_vol

    # for visualization and debug
    def build_tgt_grids(self, intrinsic, h, w, to_numpy=False):
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

        if isinstance(intrinsic, np.ndarray):
            intrinsic = torch.from_numpy(intrinsic).float()

        grids = torch.ones(3, h, w)
        grids[0, ...] = xs
        grids[1, ...] = ys

        # [3, h, w]
        camera_grid = intrinsic[:3, :3].inverse() @ grids.contiguous().view(3, -1).type_as(intrinsic)
        camera_grid = camera_grid.view(3, h, w)
        if to_numpy:
            camera_grid = camera_grid.cpu().numpy()

        return camera_grid

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx, visualize=False):
        ref_path, src_path = self.img_ids[idx]
        
        # e.g., one sample is: scene0011_00/1378.jpg;
        scene_id, ref_img_id = ref_path.split('/')
        _, src_img_id = src_path.split('/')

        h = self.cfg.INPUT.MIN_SIZE_TRAIN[0] # e.g., h == 480;
        w = int(h * 4 / 3) # e.g., then w = 640;

        ref_img_path = self.get_img_path(scene_id, ref_img_id)
        assert osp.exists(ref_img_path), f"No found {ref_img_path}"

        src_img_path = self.get_img_path(scene_id, src_img_id)
        assert osp.exists(src_img_path), f"No found {src_img_path}"

        ref_img = Image.open(ref_img_path)
        # color image in size 1296 x 968;
        # depth map in size 640 x 480;
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

        if not self.is_test_split:
            # load plane mask and generate axis-aligned bboxes and get valid indices;
            ref_seg_path = osp.join(self.seg_dir, scene_id, ref_img_id.replace('jpg', 'png'))
            if self.mode == 'mask':
                ref_seg = self.load_cleaned_seg(ref_seg_path)
                bboxes, raw_masks, ref_valid_ids = self.process_seg(ref_seg)

            else:
                raise NotImplementedError

            # if there is no valid mask, skip into the next sample
            if len(raw_masks) == 0:
                images, target, _ = self[(idx + 1) % len(self)]

                return images, target, idx

            ref_semantic_path = osp.join(self.data_root, scene_id, 
                                         self.semantic_dir, 
                                         ref_img_id.replace('jpg', 'png')
                                         )
            raw_semantic_map = self.load_semantic(ref_semantic_path, resize=(w, h))
            transformed_semantic_map = self.transform_label(raw_semantic_map)
            labels = self.get_majority_labels(transformed_semantic_map, raw_masks)

            valid_id_mask = labels > 0
            if np.sum(valid_id_mask) == 0:
                images, target, _ = self[(idx + 1) % len(self)]

                return images, target, idx

            bboxes = bboxes[valid_id_mask]
            raw_masks = raw_masks[valid_id_mask]

            target = BoxList(bboxes, ref_img.size, mode='xyxy')
            masks = SegmentationMask(raw_masks, ref_img.size, mode=self.mode)
            target.add_field('masks', masks)

            labels = labels[valid_id_mask]
            target.add_field('labels', torch.from_numpy(labels).long())

            if not self.is_train:
                planar_semantic_map = self.get_planar_semantic_map(labels, raw_masks)
                target.add_field(
                    'planar_semantic_map', 
                    torch.from_numpy(planar_semantic_map))

        else:
            # placeholder
            target = BoxList(torch.zeros(0, 4), ref_img.size, mode='xyxy')

        ref_depth_path = self.get_depth_path(scene_id, ref_img_id)
        depth = self.load_depth(ref_depth_path)
        target.add_field('depth', depth)

        # feature map size used as stereo input
        stereo_h = self.cfg.MODEL.STEREO.STEREO_H
        stereo_w = self.cfg.MODEL.STEREO.STEREO_W

        # if we do pooling for feature map, we need to further rescale the intrinsic
        if self.cfg.MODEL.STEREO.POOL_FEATURE:
            stereo_h = stereo_h // 2
            stereo_w = stereo_w // 2

        intrinsic_path = self.get_cam_intrinsic_path(scene_id)
        # intrinsic with feature map size
        intrinsic_for_stereo = self.load_intrinsic(intrinsic_path, stereo_h, stereo_w, ori_h, ori_w)
        target.add_field('intrinsic_for_stereo', torch.from_numpy(intrinsic_for_stereo).float())

        # intrinsic with image size
        intrinsic = self.load_intrinsic(intrinsic_path, h, w, ori_h, ori_w)
        target.add_field('intrinsic', torch.from_numpy(intrinsic).float())

        ref_pose_path = self.get_pose_path(scene_id, ref_img_id)
        ref_pose = self.load_pose(ref_pose_path)
        target.add_field('ref_pose', torch.from_numpy(ref_pose).float())

        src_pose_path = self.get_pose_path(scene_id, src_img_id)
        src_pose = self.load_pose(src_pose_path)
        target.add_field('src_pose', torch.from_numpy(src_pose).float())

        if self.with_instance_planar_depth_loss:
            img_tgt_camera_grids = self.build_tgt_grids(intrinsic, h, w).float()
            target.add_field('img_camera_grid', img_tgt_camera_grids)

        if not self.is_test_split:
            ref_valid_ids = ref_valid_ids[valid_id_mask]
            plane_path = osp.join(self.data_root, scene_id, self.plane_dir)
            planes = self.load_plane(plane_path, ref_valid_ids)

            # get n/d plane map
            n_div_d_map, plane_instances = \
                self.transform_normal_div_offset(planes, ref_pose, raw_masks)

            plane_mask = raw_masks.sum(0) > 0
            target.add_field('planar_mask', plane_mask)
            target.add_field('n_div_d_map', torch.from_numpy(n_div_d_map).float())

            if self.train_pixel_plane or self.with_pixel_gt_plane_map:
                pixel_plane_path = osp.join(self.pixel_plane_dir, scene_id + '_' + ref_img_id.split('.')[0] + '_' + 'plane.npy')
                pixel_plane_mask_path = osp.join(self.pixel_plane_dir, scene_id + '_' + ref_img_id.split('.')[0] + '_' + 'mask.npy')

                h, w = n_div_d_map.shape[:2]

                pixel_plane_map, pixel_plane_mask = self.load_pixel_plane(pixel_plane_path, pixel_plane_mask_path, h, w)
                plane_mask = (raw_masks.sum(0) > 0).cpu().numpy()

                # overwrite the plane map and mask computed only from plane instances
                if self.train_pixel_plane:
                    n_div_d_map = n_div_d_map * plane_mask[..., None] + pixel_plane_map * (1 - plane_mask[..., None])
                    target.add_field('n_div_d_map', torch.from_numpy(n_div_d_map).float())

                    plane_mask = ((plane_mask > 0) + (pixel_plane_mask > 0)) > 0
                    target.add_field('planar_mask', torch.from_numpy(plane_mask).float())

                elif self.with_pixel_gt_plane_map:
                    target.add_field('pixel_n_div_d_map', torch.from_numpy(pixel_plane_map).float())
                    target.add_field('pixel_plane_mask', torch.from_numpy(pixel_plane_mask).float())

            # only used for evaluation, save n / d ( where nx + d = 0)
            if not self.is_train:
                target.add_field('plane_instances', torch.from_numpy(plane_instances))

        # plane hypos and homography matrix
        homo_grid, hypos = self.build_homo_grids(ref_pose, src_pose, intrinsic_for_stereo)
        images['homo_grid'] = torch.from_numpy(homo_grid).float()
        images['hypos'] = torch.from_numpy(hypos).float()

        if not self.is_train:
            target.add_field('sem_mapping', self.id_to_label)

        if self.transforms is not None:
            processed_img, target = self.transforms(ref_img, target)
            #print_indebugmode(f"transforms applied to ref_img")
            images['ref_img'] = processed_img

            processed_img_src, _ = self.transforms(src_img, target)
            #print_indebugmode(f"transforms applied to src_img")
            images['src_img'] = processed_img_src

        return images, target, idx
