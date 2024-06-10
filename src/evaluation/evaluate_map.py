"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
import numpy as np

""" load our own modules """
from src.tools.utils import print_indebugmode


def compute_img_label_iou(pred, target, label, device='cuda'):
    gt_depth = target.get_field('depth').to(device)
    valid_mask = gt_depth > 1e-4

    pred_masks = pred.get_field('mask').to(device).squeeze(dim=1)
    gt_masks = target.get_field('masks').instances.masks.to(device)

    pred_labels = pred.get_field('labels').to(device)
    gt_labels = target.get_field('labels').to(device)

    pred_idxs = pred_labels == (label + 1)
    gt_idxs = gt_labels == (label + 1)
    
    #print_indebugmode (f"???? 0 target fields = {target.fields()}")
    pred = pred.copy_with_fields(['mask', 'scores'])
    target = target.copy_with_fields(['masks'])

    pred = pred[pred_idxs]
    #print_indebugmode (f"???? gt_idxs = {gt_idxs}, target = {target}")
    target = target[gt_idxs]

    if target.bbox.size(0) == 0 or pred.bbox.size(0) == 0:
        return [], []

    pred_masks = pred.get_field('mask').to(device).squeeze(dim=1)
    gt_masks = target.get_field('masks').instances.masks.to(device)

    pred_scores = pred.get_field('scores').to(device)
    score_rank = pred_scores.argsort(descending=True)

    pred_masks = pred_masks[score_rank]
    pred_scores = pred_scores[score_rank]

    masks_intersection = (pred_masks.unsqueeze(1) * gt_masks.unsqueeze(0) * valid_mask).float()
    masks_union = (((pred_masks.unsqueeze(1) + gt_masks.unsqueeze(0)) > 0) * valid_mask).float()

    # [nD, nG]
    ious = masks_intersection.sum(-1).sum(-1) / (masks_union.sum(-1).sum(-1) + 1e-10)
    ious = ious.cpu().numpy()

    pred_scores = pred_scores.cpu().numpy()

    return ious, pred_scores


def evaluate_img(ious, dt_scores):
    if len(ious) == 0:
        return None

    G = ious.shape[1]
    D = ious.shape[0]

    gtm = np.zeros(G)
    dtm = np.zeros(D)

    for dind in range(D):
        iou_thr = 0.5
        m = -1

        for gind in range(G):
            if gtm[gind] > 0:
                continue

            if ious[dind, gind] < iou_thr:
                continue

            iou = ious[dind, gind]
            m = gind

        if m == -1:
            continue

        dtm[dind] = 1
        gtm[m] = 1

    return {
        "dtMatches": dtm,
        'gtMatches': gtm,
        "dtScores": dt_scores
    }


def evaluate_map(pred, gt, label):
    if isinstance(pred, list):
        pred = pred[0]

    if isinstance(gt, list):
        gt = gt[0]
    
    #print_indebugmode(f"??? pred={pred.device}, gt={gt.device}, label = {label}")
    single_img_iou, single_img_score = compute_img_label_iou(pred, gt, label, device=pred.device)
    eval_res = evaluate_img(single_img_iou, single_img_score)

    return eval_res


def accumulate_map(num_K, all_eval_res):
    precision = -np.ones(num_K)
    recall = -np.ones(num_K)

    cls_aps = []

    for k in range(num_K):
        E = all_eval_res[k]
        E = [e for e in E if e is not None]

        if len(E) == 0:
            continue

        dt_scores = np.concatenate([e['dtScores'] for e in E], axis=0)
        inds = np.argsort(-dt_scores, kind='mergesort')

        dtm = np.concatenate([e['dtMatches'] for e in E], axis=0)[inds]

        tps = dtm > 0
        fps = np.logical_not(tps)

        npig = np.concatenate([e['gtMatches'] for e in E], axis=0).shape[0]

        if npig == 0:
            continue

        tp_sum = np.cumsum(tps, axis=0).astype(np.float32)
        fp_sum = np.cumsum(fps, axis=0).astype(np.float32)

        tp = np.array(tp_sum)
        fp = np.array(fp_sum)

        nd = len(tp)

        rc = tp / npig
        pr = tp / (fp + tp + 1e-4)

        if nd:
            recall[k] = np.mean(rc)

        else:
            recall[k] = 0

        for i in range(nd-1, 0, -1):
            if pr[i] > pr[i-1]:
                pr[i-1] = pr[i]

        prev_recall = 1.0
        max_precision = 0.0

        AP = 0.0

        for pr_, rc_ in zip(pr[::-1], rc[::-1]):
            AP += (prev_recall - rc_) * max_precision
            max_precision = max(max_precision, pr_)
            prev_recall = rc_

        AP += prev_recall * max_precision

        cls_aps.append(AP)

    return np.mean(cls_aps)
