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

import torch.nn.functional as F

from PIL import Image

import torch

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

""" load our own modules """
from src.config import cfg
from src.utils.colormap import ColorPalette


class ScannetDataset(torch.utils.data.Dataset):
    CLASSES = ('__background__',
               'planes'
               )

    def __init__(self,
                 data_root,
                 data_split_file,
                 cleaned_seg_dir,
                 anchor_normal_dir,
                 semantic_mapping_dir,
                 split,
                 use_new_mapping=False,
                 mode='mask',
                 transforms=None,
                 min_area=1,
                 min_h=1,
                 min_w=1,
                 mini=None):
        
        assert split in ['train', 'val', 'test']
        self.cfg = cfg
        self.split = split

        self.is_train = split == 'train'

        self.data_root = data_root
        self.anchor_normal_dir = anchor_normal_dir
        self.seg_dir = cleaned_seg_dir
        self.semantic_mapping_dir = semantic_mapping_dir

        self.use_new_mapping = use_new_mapping

        # relative directory
        self.color_dir     = 'frames/color'
        self.depth_dir     = 'frames/depth'
        self.intrinsic_dir = 'frames/intrinsic'
        self.pose_dir      = 'frames/pose'
        self.semantic_dir  = 'label-filt'
        self.plane_dir     = 'annotation/planes.npy'

        self.mode = mode
        self.transforms = transforms

        # dummy args:
        # here we clean the raw plane masks offline
        # so there is no need to constrain the min_h, min_w, min_area
        self.min_area = min_area
        self.min_h = min_h
        self.min_w = min_w

        self.data_split_file = data_split_file
        self.img_ids = self.build_list()

        self.scannet_to_nyu, self.nyu_to_canonical, self.id_to_label = self.load_mapping()

        self.colors = ColorPalette(5000).getColorMap()

        # plus 'background' class
        assert len(self.id_to_label) == self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES - 1

        if self.use_new_mapping:
            assert self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES == 16
        else:
            assert self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES == 12

    def build_list(self):
        data_paths = []
        # save as 'scene_id/img_id' format
        with open(self.data_split_file, 'r') as fp:
            for line in fp:
                data_paths.append(line.strip())

        return data_paths

    def mask_to_bbox(self, mask):
        fg = np.where(mask > 0)
        # (x1, y1, x2, y2)
        bbox = [
            np.min(fg[1]),
            np.min(fg[0]),
            np.max(fg[1]) + 1,
            np.max(fg[0]) + 1
        ]

        bbox = list(map(int, bbox))

        return bbox

    # load pre-processed segmentation map
    def load_cleaned_seg(self, seg_path):
        #print (f"??? reading {seg_path}")
        raw_seg = cv2.imread(seg_path, -1).astype(np.uint16)

        return raw_seg

    # transfer the mask into bboxes, and record the valid indexes for next filtering
    def process_seg(self, seg, invalid_ids=[65535]):
        ins_ids = np.unique(seg)

        bboxes = []
        masks = []
        valid_ids = []

        for ins_id in ins_ids:
            if ins_id in invalid_ids:
                continue

            mask = seg == ins_id
            box = self.mask_to_bbox(mask)

            bboxes.append(box)
            masks.append(mask)
            valid_ids.append(ins_id)

        bboxes = np.asarray(bboxes)
        masks = np.asarray(masks)

        h, w = seg.shape[:2]

        bboxes[:, 0::2].clip(min=0, max=w - 1)
        bboxes[:, 1::2].clip(min=0, max=h - 1)

        bboxes = torch.from_numpy(bboxes)
        masks = torch.from_numpy(masks)

        valid_ids = np.asarray(valid_ids)

        return bboxes, masks, valid_ids


    # load raw depth, and turn it into meter
    def load_depth(self, depth_path, divisor=1000.0):
        depth_map = cv2.imread(depth_path, -1).astype(np.float32) / divisor
        depth_map_t = torch.from_numpy(depth_map).float()
        return depth_map_t

    # load raw semantic map by scannet, and do resize as img size
    def load_semantic(self, semantic_path, resize):
        semantic_map = cv2.imread(semantic_path, -1)
        semantic_map = cv2.resize(semantic_map, resize, interpolation=cv2.INTER_NEAREST)

        # semantic_map = np.ones((480, 640)).astype(np.uint8)
        return semantic_map
    

    # the mapping from scannet class to nyu class, then to our class
    def load_mapping(self):
        if self.use_new_mapping:
            scannet_nyu_map = json.load(open(osp.join(self.semantic_mapping_dir, 'my_scannet_nyu_map.json')))
            nyu_canonical_map = json.load(open(osp.join(self.semantic_mapping_dir, 'my_canonical_mapping.json')))
        else:
            scannet_nyu_map = json.load(open(osp.join(self.semantic_mapping_dir, 'scannet_nyu_map.json')))
            nyu_canonical_map = json.load(open(osp.join(self.semantic_mapping_dir, 'canonical_mapping.json')))

        scannet_id_to_nyu_id = {}
        for key, val in scannet_nyu_map.items():
            scannet_id_to_nyu_id[int(key)] = int(val[-1])

        nyu_id_to_nyu_label = {}
        for key, val in scannet_nyu_map.items():
            nyu_id_to_nyu_label[int(val[1])] = val[0]

        nyu_id_to_canoical_id = {}
        canonical_id_to_label = {}

        for key, val in nyu_canonical_map.items():
            nyu_id_to_canoical_id[int(key)] = int(val)
            canonical_id_to_label[int(val)] = nyu_id_to_nyu_label[int(key)]

        return scannet_id_to_nyu_id, nyu_id_to_canoical_id, canonical_id_to_label

    # transform original scannet classes into the classes we defined
    def transform_label(self, semantic_map):
        new_semantic_map = np.zeros(semantic_map.shape)
        for label in np.unique(semantic_map):
            if label == 0:
                continue

            if label not in self.scannet_to_nyu.keys():
                continue

            else:
                nyu_label = self.scannet_to_nyu[label]
                canonical_label = self.nyu_to_canonical[nyu_label]

                mask = semantic_map == label
                new_semantic_map[mask] = canonical_label + 1

        return new_semantic_map

    # since plane mask and original semantic map does not always align,
    # we do a majority vote to decide its semantic label
    def get_majority_labels(self, semantic_map, masks):
        labels = []
        for mask in masks:
            major_label = stats.mode(semantic_map[mask], keepdims=True)[0][0]
            labels.append(major_label)

        labels = np.asarray(labels, dtype=np.uint8)

        return labels

    # pixel-level plane semantic map, only used for evaluation
    def get_planar_semantic_map(self, labels, masks):
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        h, w = masks[0].shape
        semantic_map = np.zeros((h, w)).astype(np.uint8)

        for label, mask in zip(labels, masks):
            semantic_map[mask] = label

        return semantic_map

    # load intrinsic and do scaling according to the 
    # exact image/feature size
    def load_intrinsic(self, intrinsic_path, h, w, ori_h, ori_w):
        # color image in size 1296 x 968;
        # depth map in size 640 x 480;
        
        # a 4x4 matrix
        intrinsic = np.loadtxt(intrinsic_path) # intrinsic_color;
        
        # fx and fy
        intrinsic[0][0] = intrinsic[0][0] * w / ori_w
        intrinsic[1][1] = intrinsic[1][1] * h / ori_h
        
        # here just assign cx = w/2 and cy = h/2 for simplicity;
        intrinsic[0][2] = w / 2.0 # cx
        intrinsic[1][2] = h / 2.0 # cy

        return intrinsic

    # load ctw pose, which maps points in the camera to 
    # the points in the world:
    # i.e., p^{w} = T^w_c * p_{c}
    def load_pose(self, pose_path):
        pose = np.loadtxt(pose_path)
        return pose

    # load raw planes in world coord
    def load_plane(self, plane_path, plane_idxs):
        all_planes = np.load(plane_path)

        planes = all_planes[plane_idxs]

        return planes

    def transform_plane(self, planes, pose):
        ds = np.linalg.norm(planes, axis=-1, keepdims=True)
        ns = planes / (ds + 1e-10)

        world_planes = np.concatenate([ns, -ds], axis=-1)
        camera_planes = world_planes @ pose

        # normals
        camera_normals = camera_planes[:, :3] / (np.linalg.norm(camera_planes[:, :3], axis=-1, keepdims=True) + 1e-10)

        # n / d for evaluation (nx + d) = 0
        plane_instances = camera_planes[:, :3] / (camera_planes[:, -1:] + 1e-10)

        return camera_normals, plane_instances

    # get the closest anchor normal index as its cls, and
    # compute the residual normal vector
    def get_normal_cls_res(self, camera_normals, anchor_normals):
        distance_from_anchors = camera_normals[:, None, ...] - anchor_normals[None, ...]
        normal_cls = np.linalg.norm(distance_from_anchors, axis=-1).argmin(axis=-1)

        normal_res = distance_from_anchors[np.arange(normal_cls.shape[0]), normal_cls]

        return normal_cls, normal_res

    # used for debug
    def visualize_plane(self, img, origin_masks, masks, semantic_labels, raw_semantic_map, scene_id, img_id):
        origin_ins_map = np.zeros((*origin_masks[0].shape, 3))

        ins_map = np.zeros((*masks[0].shape, 3))
        sem_map = np.zeros((*masks[0].shape, 3))

        raw_sem_vis = np.zeros((*raw_semantic_map.shape, 3))

        img = np.asarray(img)[..., ::-1]

        for plane_idx, mask in enumerate(origin_masks):
            origin_ins_map[mask] = self.colors[plane_idx]

        origin_ins_map = origin_ins_map * 0.5 + img * 0.5

        for plane_idx, (mask, label) in enumerate(zip(masks, semantic_labels)):
            ins_map[mask] = self.colors[plane_idx]
            sem_map[mask] = self.colors[label]

        ins_map = ins_map * 0.5 + img * 0.5
        sem_map = sem_map * 0.5 + img * 0.5

        for label in np.unique(raw_semantic_map):
            if label == 0:
                continue

            mask = raw_semantic_map == label
            raw_sem_vis[mask] = self.colors[label]

        raw_sem_vis = raw_sem_vis * 0.5 + img * 0.5

        vis = np.hstack([origin_ins_map, ins_map, sem_map, raw_sem_vis])

        save_folder = 'debug_vis'

        if not osp.exists(save_folder):
            os.makedirs(save_folder)

        cv2.imwrite(osp.join(save_folder, scene_id + '_' + img_id), vis)

        return

    def __len__(self):
        return len(self.img_ids)
    
    def get_img_path(self, scene_id, img_id):
        img_path = osp.join(self.data_root, scene_id, self.color_dir, img_id)
        return img_path 
    
    def get_depth_path(self, scene_id, img_id):
        depth_path = osp.join(self.data_root, scene_id, self.depth_dir, img_id.replace('jpg', 'png'))
        return depth_path 

    def get_cam_intrinsic_path(self, scene_id):
        intrinsic_path = osp.join(self.data_root, scene_id, self.intrinsic_dir, 'intrinsic_color.txt')
        return intrinsic_path
    
    def get_pose_path(self, scene_id, img_id):
        pose_path = osp.join(self.data_root, scene_id, self.pose_dir, img_id.replace('jpg', 'txt'))
        return pose_path

    def __getitem__(self, idx, visualize=False):
        scene_id, img_id = self.img_ids[idx].split('/')

        # get h, w from config, which is fixed in our experiment (640, 480)
        h = self.cfg.INPUT.MIN_SIZE_TRAIN[0]
        w = int(h * 4 / 3)

        img_path = self.get_img_path(scene_id, img_id)
        if not osp.exists(img_path):
            images, target, _ = self[(idx + 1) % len(self)]

            return images, target, idx

        img = Image.open(img_path)
        ori_w, ori_h = img.size
        img = img.resize((w, h)).convert('RGB')

        images = {}

        if not self.is_train:
            images['ori_ref_img'] = np.asarray(img)

        else:
            images['ori_ref_img'] = img

        images['ref_path'] = self.img_ids[idx]

        # load plane mask and generate axis-aligned bboxes and get valid indices;
        seg_path = osp.join(self.seg_dir, scene_id, img_id.replace('jpg', 'png'))
        if self.mode == 'mask':
            seg = self.load_cleaned_seg(seg_path)
            bboxes, raw_masks, valid_ids = self.process_seg(seg)

        else:
            raise NotImplementedError

        # if there is no valid mask, skip into the next sample
        if len(raw_masks) == 0:
            images, target, _ = self[(idx + 1) % len(self)]

            return images, target, idx

        origin_raw_masks = raw_masks.clone()

        # get semantic labels for each plane by majority-voting
        semantic_path = osp.join(self.data_root, scene_id, self.semantic_dir, img_id.replace('jpg', 'png'))
        raw_semantic_map = self.load_semantic(semantic_path, resize=(w, h))
        transformed_semantic_map = self.transform_label(raw_semantic_map)
        labels = self.get_majority_labels(transformed_semantic_map, raw_masks)

        # get binary mask for the valid planes, to filter the invalid ones
        valid_id_mask = labels > 0
        if np.sum(valid_id_mask) == 0:
            images, target, _ = self[(idx + 1) % len(self)]

            return images, target, idx

        # filter
        bboxes = bboxes[valid_id_mask]
        raw_masks = raw_masks[valid_id_mask]

        # put all the labels into "BoxList"
        target = BoxList(bboxes, img.size, mode='xyxy')
        masks = SegmentationMask(raw_masks, img.size, mode=self.mode)
        target.add_field('masks', masks)

        labels = labels[valid_id_mask]
        target.add_field('labels', torch.from_numpy(labels).long())

        # use for plane semantic evaluation(i.e., m-iou...)
        if not self.is_train:
            planar_semantic_map = self.get_planar_semantic_map(labels, raw_masks)
            target.add_field('planar_semantic_map', torch.from_numpy(planar_semantic_map))

        depth_path = self.get_depth_path(scene_id, img_id)
        depth = self.load_depth(depth_path)

        target.add_field('depth', depth)

        # load intrinsic and resize it
        intrinsic_path = self.get_cam_intrinsic_path(scene_id)
        intrinsic = self.load_intrinsic(intrinsic_path, h, w, ori_h, ori_w)
        target.add_field('intrinsic', torch.from_numpy(intrinsic))

        pose_path = self.get_pose_path(scene_id, img_id)
        pose = self.load_pose(pose_path)

        # load planes and transform to camera view
        valid_ids = valid_ids[valid_id_mask]
        plane_path = osp.join(self.data_root, scene_id, self.plane_dir)
        planes = self.load_plane(plane_path, valid_ids)

        camera_normals, plane_instances = self.transform_plane(planes, pose)
        if not self.is_train:
            target.add_field('plane_instances', torch.from_numpy(plane_instances))

        # anchor normals by K-Means clustering
        anchor_normals = np.load(self.anchor_normal_dir)
        normal_cls, normal_res = self.get_normal_cls_res(camera_normals, anchor_normals)

        target.add_field('pose', torch.from_numpy(pose))

        target.add_field('anchor_normals', torch.from_numpy(anchor_normals))
        target.add_field('normal_cls', torch.from_numpy(normal_cls))
        target.add_field('normal_res', torch.from_numpy(normal_res))

        if not self.is_train:
            target.add_field('sem_mapping', self.id_to_label)

        # visualize the instance and semantic map for planes, just used for debug
        if visualize:
            self.visualize_plane(img, origin_raw_masks, raw_masks, labels, raw_semantic_map, scene_id, img_id)

        # do pre-process for image data(augmentation, if necessary)
        if self.transforms is not None:
            processed_img, target = self.transforms(img, target)

            images['ref_img'] = processed_img

        return images, target, idx
