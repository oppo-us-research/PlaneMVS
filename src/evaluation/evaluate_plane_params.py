"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
import numpy as np

""" load our own modules """
from src.tools.utils import print_indebugmode


def evaluate_plane_params(pred_planes, target, pos_pred_idxs, matched_gt_idxs):
    # both of them represents: nx + d = 0
    if pos_pred_idxs is None or pos_pred_idxs.sum() == 0:
        return None, None, None

    pos_pred_planes = pred_planes[pos_pred_idxs]
    gt_planes = target.get_field('plane_instances')

    matched_gt_planes = gt_planes[matched_gt_idxs][pos_pred_idxs]

    pos_pred_planes = pos_pred_planes.cpu().numpy()
    matched_gt_planes = matched_gt_planes.cpu().numpy() #[n, 3]
    #print_indebugmode(f"matched_gt_planes = {matched_gt_planes.shape}, pos_pred_planes = {pos_pred_planes.shape}")


    pred_n = pos_pred_planes / (1.0e-8+ np.linalg.norm(pos_pred_planes,axis=-1))[..., None]
    pred_d = 1 / (1.0e-8 + np.linalg.norm(pos_pred_planes, axis=-1))[..., None]

    # there are some invalid gt plane params, whose offset is 0
    valid_gt_idxs = np.linalg.norm(matched_gt_planes, axis=-1) > 1e-4
    if valid_gt_idxs.sum() == 0:
        return None, None, None

    gt_n = matched_gt_planes / (1.0e-8 + np.linalg.norm(matched_gt_planes, axis=-1))[..., None]
    gt_d = 1 / (1.0e-8 + np.linalg.norm(matched_gt_planes, axis=-1))[..., None]

    # only compute those planes whose gt are valid
    pred_n = pred_n[valid_gt_idxs]
    pred_d = pred_d[valid_gt_idxs]

    gt_n = gt_n[valid_gt_idxs]
    gt_d = gt_d[valid_gt_idxs]

    normal_diff = np.mean(np.linalg.norm(pred_n - gt_n, axis=-1))
    offset_diff = np.mean(np.linalg.norm(pred_d - gt_d, axis=-1))

    pos_pred_planes = pos_pred_planes[valid_gt_idxs]
    matched_gt_planes = matched_gt_planes[valid_gt_idxs]
    n_div_d_diff = np.mean(np.linalg.norm(pos_pred_planes - matched_gt_planes, axis=-1))

    return normal_diff, offset_diff, n_div_d_diff
