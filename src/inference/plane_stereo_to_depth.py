"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
import torch.nn.functional as F

import numpy as np
import cv2

from src.tools.utils import print_indebugmode

def make_camera_grid(intrinsic, h, w, depth=None):
    # indexing added for Torch >=1.10;
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

    grid = torch.ones(3, h, w)
    grid[0, ...] = xs
    grid[1, ...] = ys

    camera_grid = intrinsic.inverse() @ grid.contiguous().view(3, -1).type_as(intrinsic)
    camera_grid = camera_grid.view(3, h, w).to('cuda')

    if depth is not None:
        points = camera_grid * depth

    else:
        points = None

    return grid, camera_grid, points


def get_pred_homography(pose_tgt, pose_src, intrinsic, n_div_d_map):
    rot_tgt = pose_tgt[:3, :3]
    trans_tgt = pose_tgt[:3, -1:]

    rot_src = pose_src[:3, :3]
    trans_src = pose_src[:3, -1:]

    n_div_d_map = n_div_d_map.permute(1, 2, 0).view(-1, 3).unsqueeze(dim=1)
    n_div_d_map = n_div_d_map.to(rot_src.device)

    homo_map = rot_src.T @ rot_tgt + (rot_src.T @ (trans_src - trans_tgt) @ n_div_d_map)
    homo_map = intrinsic @ homo_map @ intrinsic.inverse()

    return homo_map


def get_src_coords(tgt_grids, homo_map):
    tgt_grids = tgt_grids.permute(1, 2, 0).type_as(homo_map)
    src_coords = homo_map.view(-1, 3, 3) @ tgt_grids.view(-1, 3).unsqueeze(dim=-1)
    src_coords = src_coords.squeeze(dim=-1)
    src_xys = src_coords[:, :2] / src_coords[:, -1:]

    h, w = tgt_grids.size()[:2]
    src_xys = src_xys.view(h, w, 2).unsqueeze(dim=0)

    workspace = np.asarray([(w-1) / 2, (h-1) / 2])[None, None, :]
    workspace = torch.from_numpy(workspace)

    normalized_src_xys = src_xys / workspace - 1

    return normalized_src_xys



def get_instance_n_div_d(plane_map, masks):
    n_div_ds = []
    plane_instances = []

    for mask in masks:
        n_div_ds.append(torch.mean(plane_map[:, mask], dim=-1))
        plane_instances.append(torch.mean(-plane_map[:, mask], dim=-1))

    n_div_ds = torch.stack(n_div_ds)
    plane_instances = torch.stack(plane_instances)

    return n_div_ds, plane_instances


def get_soft_instance_n_div_d(plane_map, soft_masks):
    n_div_ds = []
    plane_instances = []

    soft_masks = soft_masks.to(plane_map.device)

    for mask in soft_masks:
        hard_mask = mask > 0.5

        n_div_ds.append((plane_map[:, hard_mask] * mask[hard_mask]).sum(dim=-1) / mask[hard_mask].sum())
        plane_instances.append((-plane_map[:, hard_mask] * mask[hard_mask]).sum(dim=-1) / mask[hard_mask].sum())

    n_div_ds = torch.stack(n_div_ds)
    plane_instances = torch.stack(plane_instances)

    return n_div_ds, plane_instances


def ins_plane_to_depth(ins_planes, camera_grid, masks):
    h, w = masks[0].size()[-2:]
    #print_indebugmode(f"ins_planes={ins_planes.dtype}, camera_grid={camera_grid.dtype}")
    divider = ins_planes @ camera_grid.view(3, -1)
    divider[torch.abs(divider) < 1e-4] = 1e-4

    plane_piece_depths = (1 / divider).view(masks.size(0), h, w)
    plane_piece_depths = plane_piece_depths * masks

    plane_piece_depths[plane_piece_depths < 1e-4] = 1e4
    plane_depth_map = torch.min(plane_piece_depths, dim=0)[0]

    pred_plane_mask = masks.sum(0) > 0
    plane_depth_map = plane_depth_map * pred_plane_mask

    return plane_depth_map, pred_plane_mask


def plane_stereo_to_depth(result, images, target, max_depth=10, use_refine=False, to_numpy=True, instance_pooling=True):
    assert len(result) == 1, "Currently only support batch_size=1 during testing"
    if use_refine:
        pred_n_div_d = result[0].get_field('refined_plane_map').squeeze(dim=0)

    else:
        pred_n_div_d = result[0].get_field('plane_map').squeeze(dim=0)

    if result[0].has_field('depth'):
        np_depth = result[0].get_field('depth').squeeze()

    else:
        np_depth = None

    masks = result[0].get_field('mask').squeeze(dim=1).to('cuda')
    #print_indebugmode (f'??? target fileds = {target.fields()}')
    if target.has_field('masks'):
        gt_masks = target.get_field('masks').instances.masks.to('cuda')
        gt_plane_mask = np.sum(target.get_field('masks').instances.masks.cpu().numpy(), axis=0) > 0
        target.add_field('gt_plane_mask', gt_plane_mask)

    else:
        gt_masks = None

    intrinsic = target.get_field('intrinsic')
    intrinsic = intrinsic[:3, :3]
    h, w = pred_n_div_d.size()[-2:]

    grid, camera_grid, _ = make_camera_grid(intrinsic, h, w)
    pixel_plane_depth_map = (1 / torch.sum(-pred_n_div_d * camera_grid, axis=0)).view(h, w)
    pixel_plane_depth_map = pixel_plane_depth_map.clamp(min=0, max=max_depth)

    if masks.size(0) == 0 and np_depth is not None:
        merged_depth = np_depth
        if to_numpy:
            merged_depth = merged_depth.cpu().numpy()
            pixel_plane_depth_map = pixel_plane_depth_map.cpu().numpy()
            pred_plane_mask = np.zeros(np_depth.shape).astype(np.bool)

        result[0].add_field('pred_plane_mask', pred_plane_mask)
        result[0].add_field('camera_grid', camera_grid)

        return merged_depth, pixel_plane_depth_map, pixel_plane_depth_map, [], None

    if instance_pooling and masks.size(0) > 0:
        if result[0].has_field('soft_mask'):
            soft_masks = result[0].get_field('soft_mask').squeeze(dim=1)
            ins_n_div_ds, plane_instances = get_soft_instance_n_div_d(-pred_n_div_d, soft_masks)

        else:
            ins_n_div_ds, plane_instances = get_instance_n_div_d(-pred_n_div_d, masks)

        plane_depth_map, pred_plane_mask = ins_plane_to_depth(ins_n_div_ds, camera_grid, masks)
        instance_planar_depth = plane_depth_map * pred_plane_mask + pixel_plane_depth_map * (~pred_plane_mask)

        if gt_masks is not None:
            gt_mask_ins_n_div_ds, _ = get_instance_n_div_d(-pred_n_div_d, gt_masks)
            gt_mask_plane_depth_map, gt_plane_mask = ins_plane_to_depth(gt_mask_ins_n_div_ds, camera_grid, gt_masks)

            gt_mask_depth_map = gt_mask_plane_depth_map * gt_plane_mask + pixel_plane_depth_map * (~gt_plane_mask)

        else:
            gt_mask_depth_map = None

    else:
        if masks.size(0) == 0:
            pred_plane_mask = torch.zeros(h, w, device=pred_n_div_d.device).bool()

        else:
            pred_plane_mask = masks.sum(0) > 0

        instance_planar_depth = pixel_plane_depth_map
        plane_instances = []

        if gt_masks is not None:
            gt_mask_depth_map = pixel_plane_depth_map

        else:
            gt_mask_depth_map = None

    if np_depth is not None:
        merged_depth = plane_depth_map * pred_plane_mask + np_depth * (~pred_plane_mask)
    else:
        merged_depth = None

    if to_numpy:
        if merged_depth is not None:
            merged_depth = merged_depth.cpu().numpy()

        pred_plane_mask = pred_plane_mask.cpu().numpy()
        instance_planar_depth = instance_planar_depth.cpu().numpy()
        pixel_plane_depth_map = pixel_plane_depth_map.cpu().numpy()

        if gt_mask_depth_map is not None:
            gt_mask_depth_map = gt_mask_depth_map.cpu().numpy()

    result[0].add_field('pred_plane_mask', pred_plane_mask)
    result[0].add_field('camera_grid', camera_grid)
    result[0].add_field('pixel_planar_depth', pixel_plane_depth_map)

    return merged_depth, instance_planar_depth, pixel_plane_depth_map, plane_instances, gt_mask_depth_map
