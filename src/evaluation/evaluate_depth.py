"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

import numpy as np


def evaluate_depth(gt, pred, valid_mask=None, max_depth=10):
    if not isinstance(gt, np.ndarray):
        gt = gt.squeeze(dim=0).cpu().numpy()

    if not isinstance(pred, np.ndarray):
        pred = pred.squeeze(dim=0).cpu().numpy()

    pred = pred.clip(min=1e-4, max=max_depth)

    if valid_mask is None:
        valid_mask = gt > 1e-4

    else:
        valid_mask = (gt > 1e-4) * valid_mask

    if valid_mask.sum() == 0:
        return None

    gt = gt[valid_mask]
    pred = pred[valid_mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2)

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3
