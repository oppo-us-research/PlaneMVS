"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

import numpy as np
import torch

def evaluate_plane_geometric(pred_depth, camera_grids, target):
    gt_masks = target.get_field('masks')
    gt_masks = gt_masks.instances.masks.cpu().numpy()

    camera_grids = camera_grids.cpu().numpy()

    gt_depth = target.get_field('depth').cpu().numpy()

    pred_points = camera_grids * pred_depth
    gt_points = camera_grids * gt_depth

    depth_valid_mask = gt_depth > 1e-4

    if gt_masks.shape[0] == 0:
        return None, None, None

    pred_params = []
    gt_params = []
    areas = []

    for idx, points in enumerate([pred_points, gt_points]):
        for mask in gt_masks:
            # only consider the region with valid depth
            mask = mask * depth_valid_mask
            A = points[:, mask].T
            # inv_AA = np.linalg.inv(A.T @ A)

            # Ab = A.T @ np.ones((A.shape[0], 1))

            # plane_param = inv_AA @ Ab
            # plane_offset = np.linalg.norm(plane_param, axis=0)

            # plane_param = plane_param / (np.power(plane_offset, 2) + 1e-4)
            # plane_param = plane_param.squeeze()

            lst_plane_param = np.linalg.lstsq(A, np.ones((A.shape[0], 1)), rcond=-1)[0]
            lst_plane_param = lst_plane_param.squeeze()

            if idx == 0:
                pred_params.append(lst_plane_param)
                areas.append(mask.sum())

            elif idx == 1:
                gt_params.append(lst_plane_param)

    pred_params = np.asarray(pred_params)
    gt_params = np.asarray(gt_params)

    param_diff = np.linalg.norm(pred_params - gt_params, axis=-1)

    total_diff = 0
    total_area = 0

    for pred, gt, area in zip(pred_params, gt_params, areas):
        weighted_diff = np.linalg.norm(pred - gt) * area

        total_diff += weighted_diff
        total_area += area

    return param_diff, total_diff, total_area
