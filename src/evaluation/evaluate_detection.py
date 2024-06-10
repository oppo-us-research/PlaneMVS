"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
import numpy as np


def evaluate_detection(pred, target, use_refine=False, device='cuda'):
    assert len(pred) == 1, 'Currently only support eval when batch_size=1'
    pred = pred[0]

    if use_refine:
        pred_masks = pred.get_field('refined_mask').to(device)

    else:
        pred_masks = pred.get_field('mask').to(device)

    pred_masks = pred_masks.squeeze(dim=1)

    if pred.has_field('final_scores'):
        pred_scores = pred.get_field('final_scores').to(device)

    else:
        pred_scores = pred.get_field('scores').to(device)

    if pred.has_field('planar_depth'):
        pred_depth = pred.get_field('planar_depth').to(device)

    else:
        pred_depth = pred.get_field('depth').to(device)

    score_rank = pred_scores.argsort(descending=True)
    pred_masks = pred_masks[score_rank]

    gt_depth = target.get_field('depth').to(device)
    gt_masks = target.get_field('masks').instances.masks.to(device)

    pixel_curves = []
    plane_curves = []

    if gt_masks.size(0) == 0:
        return None, None, None, None, None

    if pred_masks.size(0) == 0:
        APs = [0] * 5

        pixel_curves.append([0] * 22)
        plane_curves.append([0] * 22)

        return APs, pixel_curves, plane_curves, None, None

    valid_mask = gt_depth > 1e-4
    plane_areas = gt_masks.sum(-1).sum(-1).float()
    num_plane_pixels = plane_areas.sum()

    masks_intersection = (gt_masks.unsqueeze(1) * pred_masks.unsqueeze(0) * valid_mask).float()
    masks_union = (((gt_masks.unsqueeze(1) + pred_masks.unsqueeze(0)) > 0) * valid_mask).float()

    iou = masks_intersection.sum(-1).sum(-1) / (masks_union.sum(-1).sum(-1) + 1e-10)

    intersection_areas = masks_intersection.sum(-1).sum(-1)

    depth_diff = torch.abs(gt_depth - pred_depth)
    depth_diff[~valid_mask] = 0

    depths_diff = (depth_diff * masks_intersection).sum(-1).sum(-1) / intersection_areas.clamp(min=1e-4)
    depths_diff[intersection_areas < 1e-4] = 1e10

    pos_preds = torch.max(iou, dim=0)[0] > 0.5
    matched_gts = iou.argmax(dim=0)

    for IOU_threshold in [0.5, ]:
        iou_mask = (iou > IOU_threshold).float()
        min_diff = torch.min(depths_diff * iou_mask + 1e6 * (1 - iou_mask), dim=1)[0]
        stride = 0.05

        plane_recall = []
        pixel_recall = []

        for step in range(22):
            if step == 21:
                diff_threshold = 100

            else:
                diff_threshold = step * stride

            pixel_recall.append((torch.min((intersection_areas * ((depths_diff <= diff_threshold).float() * iou_mask)).sum(1), plane_areas).sum() / num_plane_pixels).item())
            plane_recall.append((((min_diff <= diff_threshold).float().sum()) / gt_masks.size(0)).item())

        pixel_curves.append(pixel_recall)
        plane_curves.append(plane_recall)

    APs = []

    for diff_threshold in [0.2, 0.4, 0.6, 0.9, 100]:
        correct_mask = torch.min((depths_diff < diff_threshold), (iou > 0.5))
        match_mask = torch.zeros(correct_mask.size(0)).bool().to(device)

        recalls = []
        precisions = []

        num_predictions = correct_mask.size(1)
        num_targets = (plane_areas > 0).sum().item()

        # since we have sorted in high to low...
        for rank in range(num_predictions):
            match_mask = torch.max(match_mask, correct_mask[:, rank])
            num_matches = match_mask.sum().item()

            # precision from high to low, recall from low to high
            precisions.append(float(num_matches / (rank + 1)))
            recalls.append(float(num_matches / num_targets))

        max_precision = 0.0
        prev_recall = 1.0

        AP = 0.0

        # reverse order and get AP
        for recall, precision in zip(recalls[::-1], precisions[::-1]):
            AP += (prev_recall - recall) * max_precision
            max_precision = max(max_precision, precision)
            prev_recall = recall

        AP += prev_recall * max_precision

        APs.append(AP)

    return APs, pixel_curves, plane_curves, pos_preds, matched_gts
