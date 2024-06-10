"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
import numpy as np

import torch.nn.functional as F


def evaluate_masks(pred, target, use_refine=False, default_size=(480, 640), device='cuda'):
    assert len(pred) == 1
    pred = pred[0]

    if use_refine:
        pred_masks = pred.get_field('refined_mask')

    else:
        pred_masks = pred.get_field('mask')

    pred_masks = pred_masks.squeeze(dim=1).bool().to(device)
    valid_mask = (target.get_field('depth') > 1e-4).to(device)
    gt_masks = target.get_field('masks').instances.masks.to(device)

    # concat non-planar
    gt_masks = torch.cat([gt_masks, (1 - gt_masks.sum(0, keepdim=True)).clamp(min=0).bool()], dim=0)
    pred_masks = torch.cat([pred_masks, (1 - pred_masks.sum(0, keepdim=True)).clamp(min=0).bool()], dim=0)

    if pred_masks.size(0) == 1:
        pred_masks = F.interpolate(pred_masks.unsqueeze(dim=1).float(), default_size, mode='bilinear', align_corners=True).squeeze(dim=1).bool()

    intersection = (gt_masks.unsqueeze(dim=1) * pred_masks.unsqueeze(dim=0) * valid_mask).sum(-1).sum(-1).float()
    union = (torch.max(gt_masks.unsqueeze(dim=1), pred_masks.unsqueeze(dim=0)) * valid_mask).sum(-1).sum(-1).float()

    N = intersection.sum()

    RI = 1 - ((intersection.sum(0).pow(2).sum() + intersection.sum(1).pow(2).sum()) / 2 - intersection.pow(2).sum()) / (N * (N - 1) / 2)
    joint = intersection / N
    marginal_2 = joint.sum(0)
    marginal_1 = joint.sum(1)
    H_1 = (-marginal_1 * torch.log2(marginal_1 + (marginal_1 == 0).float())).sum()
    H_2 = (-marginal_2 * torch.log2(marginal_2 + (marginal_2 == 0).float())).sum()

    B = (marginal_1.unsqueeze(-1) * marginal_2)
    log2_quotient = torch.log2(torch.clamp(joint, 1e-8) / torch.clamp(B, 1e-8)) * (torch.min(joint, B) > 1e-8).float()
    MI = (joint * log2_quotient).sum()
    voi = H_1 + H_2 - 2 * MI

    IOU = intersection / torch.clamp(union, min=1)
    SC = ((IOU.max(-1)[0] * torch.clamp((gt_masks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N + (IOU.max(0)[0] * torch.clamp((pred_masks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N) / 2

    metrics = np.asarray([voi.cpu().numpy(), RI.cpu().numpy(), SC.cpu().numpy()])

    return metrics
