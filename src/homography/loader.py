"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import os.path as osp

import tqdm

import numpy as np
import cv2

import torch
import torch.nn.functional as F




def load_img(img_path, resize=(640, 480)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, resize)

    return img


def load_plane(plane_path, valid_ids):
    raw_planes = np.load(plane_path)
    valid_planes = raw_planes[valid_ids]

    return valid_planes


def load_intrinsic(intrinsic_path):
    intrinsic = np.loadtxt(intrinsic_path)

    return intrinsic


def load_pose(pose_path):
    pose = np.loadtxt(pose_path)

    return pose


def load_seg(seg_path):
    seg = cv2.imread(seg_path, -1)
    valid_ids = [seg_id for seg_id in np.unique(seg) if seg_id != 65535]

    masks = []
    for v_id in valid_ids:
        masks.append(seg == v_id)

    masks = np.asarray(masks)
    valid_ids = np.asarray(valid_ids)

    return masks, valid_ids


def transform_plane(raw_planes, pose):
    ds = np.linalg.norm(raw_planes, axis=-1, keepdims=True)
    ns = raw_planes / (ds + 1e-10)

    world_planes = np.concatenate([ns, -ds], axis=-1)
    camera_planes = world_planes @ pose

    return camera_planes


def get_normal_map(plane_tgt, masks):
    normal_map = np.ones((*masks[0].shape, 3))
    ns = plane_tgt[:, :3]

    for mask, n in zip(masks, ns):
        normal_map[mask, :] = n

    return normal_map


def get_offset_map(plane_tgt, masks):
    offset_map = np.ones(masks[0].shape)
    ds = plane_tgt[:, -1:]

    for mask, d in zip(masks, ds):
        offset_map[mask] = d

    return offset_map


def get_homography(pose_tgt, pose_src, plane_tgt, intrinsic, masks):
    rot_tgt = pose_tgt[:3, :3]
    trans_tgt = pose_tgt[:3, -1:]

    rot_src = pose_src[:3, :3]
    trans_src = pose_src[:3, -1:]

    normal_map = get_normal_map(plane_tgt, masks)
    offset_map = get_offset_map(plane_tgt, masks)

    h, w = normal_map.shape[:2]

    normal_map = normal_map.reshape(-1, 3)[:, None, :]
    offset_map = offset_map.reshape(-1, 1)[:, None, :]

    # (3, 3) @ (3, 1) @ (n, 1, 3) / (n, 1, 1) -> (n, 3, 3)
    homo = rot_src.T @ rot_tgt + (rot_src.T @ (trans_src - trans_tgt)) @ normal_map / (offset_map + 1e-10)
    # (n, 3, 3)
    homo = intrinsic[:3, :3] @ homo @ np.linalg.inv(intrinsic[:3, :3])
    homo = homo.reshape(h, w, 3, 3)

    return homo


def make_grids(h=480, w=640):
    xxs, yys = np.meshgrid(np.arange(w), np.arange(h))
    xys = np.ones((3, h, w))
    xys[0, ...] = xxs
    xys[1, ...] = yys

    xys = xys.transpose(1, 2, 0)

    return xys


def get_src_coords(grids, homo):
    # (n, 3, 3) @ (n, 3)
    src_coords = homo.reshape(-1, 3, 3) @ grids.reshape(-1, 3)[..., None]
    src_coords = src_coords.squeeze(axis=-1)
    src_xys = src_coords[:, :2] / src_coords[:, -1:]

    h, w = grids.shape[:2]
    src_xys = src_xys.reshape(h, w, 2)

    return src_xys


def warp_by_homography(homo, src_img):
    target_size = (src_img.shape[1], src_img.shape[0])
    warped_img = cv2.warpPerspective(src_img, homo, target_size)

    return warped_img


def warp_src_torch(src_grids, src_img):
    src_img = torch.from_numpy(src_img).permute(2, 0, 1).unsqueeze(dim=0)
    src_grids = torch.from_numpy(src_grids).unsqueeze(dim=0)

    src_img = src_img.type_as(src_grids)

    warped_src_img = F.grid_sample(src_img, src_grids, mode='bilinear', padding_mode='zeros')
    warped_src_img = warped_src_img.squeeze(dim=0).permute(1, 2, 0).cpu().numpy()

    return warped_src_img


def load_stereo_pairs(stereo_file):
    paths = []
    with open(stereo_file, 'r') as fp:
        for line in fp:
            ref_path, src_path = line.strip().split('\t')
            paths.append([ref_path, src_path])

    return paths


def main():
    stereo_file = '../data_preparation/scannet_splits/stereo_val_files.txt'
    data_dir = '/mnt/Data/jiachenliu/scannet_data/scans'
    seg_dir = '../data_preparation/stereo_cleaned_segmentation'
    vis_save_dir = 'pos_d_homo_warp_vis'
    
    stereo_pair_paths = load_stereo_pairs(stereo_file)

    tgt_grids = make_grids()

    for stereo_pair in tqdm.tqdm(stereo_pair_paths):
        ref_path, src_path = stereo_pair
        scene, ref_img_id = ref_path.split('/')
        src_img_id = src_path.split('/')[-1]

        src_path = osp.join(data_dir, scene, 'frames/color', src_img_id)
        ref_path = osp.join(data_dir, scene, 'frames/color', ref_img_id)

        src_img = load_img(src_path)
        ref_img = load_img(ref_path)

        seg_path = osp.join(seg_dir, scene, ref_img_id.replace('jpg', 'png'))
        if not osp.exists(seg_path):
            continue

        ref_masks, ref_plane_ids = load_seg(seg_path)

        valid_plane_mask = ref_masks.sum(axis=0) > 0

        plane_path = osp.join(data_dir, scene, 'annotation/planes.npy')
        planes = load_plane(plane_path, ref_plane_ids)

        intrinsic_path = osp.join(data_dir, scene, 'frames/intrinsic', 'intrinsic_depth.txt')
        intrinsic = load_intrinsic(intrinsic_path)

        ref_pose_path = osp.join(data_dir, scene, 'frames/pose', ref_img_id.replace('jpg', 'txt'))
        ref_pose = load_pose(ref_pose_path)

        src_pose_path = osp.join(data_dir, scene, 'frames/pose', src_img_id.replace('jpg', 'txt'))
        src_pose = load_pose(src_pose_path)

        camera_planes = transform_plane(planes, ref_pose)

        homo = get_homography(ref_pose, src_pose, camera_planes, intrinsic, ref_masks)
        src_grids = get_src_coords(tgt_grids, homo)

        h, w = src_grids.shape[:2]

        workspace = np.asarray([(w - 1) / 2, (h - 1) / 2])[None, None, :]

        src_grids = src_grids / workspace - 1

        warped_src_img = warp_src_torch(src_grids, src_img)
        warped_src_img = warped_src_img * valid_plane_mask[..., None]
        vis = np.hstack([ref_img, src_img, warped_src_img])
        vis = cv2.resize(vis, None, fx=0.7, fy=0.7)

        if not osp.exists(vis_save_dir):
            os.makedirs(vis_save_dir)

        save_name = scene + '_' + ref_img_id.split('.')[0] + '_' + src_img_id.split('.')[0] + '.jpg'
        cv2.imwrite(osp.join(vis_save_dir, save_name), vis)

if __name__ == '__main__':
    main()
