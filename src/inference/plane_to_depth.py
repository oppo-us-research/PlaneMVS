"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
import numpy as np


def make_camera_grid(intrinsic, depth):
    h, w = depth.size()
    # indexing added for Torch >=1.10;
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

    grid = torch.ones(3, h, w)
    grid[0, ...] = xs
    grid[1, ...] = ys

    grid = intrinsic.inverse() @ grid.contiguous().view(3, -1).type_as(intrinsic)
    grid = grid.view(3, h, w).to(depth.device)

    points = grid * depth

    return grid, points


def plane_to_depth(result, target, max_depth=10, to_numpy=True):
    assert len(result) == 1, "Currently only support batch_size=1 during testing"
    np_depth = result[0].get_field('depth').squeeze()

    masks = result[0].get_field('mask').squeeze(dim=1).to(np_depth.device)

    if 'normal_res' not in result[0].fields():
        depth = np_depth
        pred_plane_mask = masks.sum(0) > 0

        if to_numpy:
            depth = depth.clamp(min=0, max=max_depth).cpu().numpy()
            pred_plane_mask = pred_plane_mask.cpu().numpy()

        return depth, pred_plane_mask, None

    if masks.size(0) == 0:
        depth = np_depth
        if to_numpy:
            depth = depth.clamp(min=0, max=max_depth).cpu().numpy()
            pred_plane_mask = np.zeros(depth.shape).astype(np.bool)

        return depth, pred_plane_mask, None

    h, w = masks.size()[-2:]

    intrinsic = target.get_field('intrinsic')
    intrinsic = intrinsic[:3, :3]

    camera_grid, points = make_camera_grid(intrinsic, np_depth)

    pred_normals = result[0].get_field('normal_res')
    pred_normals = pred_normals / (torch.norm(pred_normals, dim=-1, keepdim=True) + 1e-10)
    offsets = ((pred_normals @ points.view(3, -1)).view(masks.size(0), h, w) * masks).sum(dim=-1).sum(dim=-1) / masks.sum(dim=-1).sum(dim=-1)

    plane_piece_depths = offsets.unsqueeze(dim=-1).unsqueeze(dim=-1) / ((pred_normals @ camera_grid.view(3, -1)).view(masks.size(0), h, w) + 1e-10)

    plane_piece_depths = plane_piece_depths * masks

    plane_piece_depths[plane_piece_depths < 1e-4] = 1e4
    plane_depth_map = torch.min(plane_piece_depths, dim=0)[0]

    pred_plane_mask = masks.sum(0) > 0
    plane_depth_map = plane_depth_map * pred_plane_mask

    depth = plane_depth_map * pred_plane_mask + np_depth * (~pred_plane_mask)

    depth = depth.clamp(min=0, max=max_depth)

    if to_numpy:
        depth = depth.cpu().numpy()
        pred_plane_mask = pred_plane_mask.cpu().numpy()

    # n / d for evaluation (nx + d = 0)
    plane_instances = pred_normals / (-offsets.unsqueeze(dim=-1))

    return depth, pred_plane_mask, plane_instances
