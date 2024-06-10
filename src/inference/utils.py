"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import numpy as np
import torch


def get_planar_semantic_map(labels, masks):
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    masks = masks.squeeze(axis=1)

    if masks.shape[0] == 0:
        semantic_map = np.zeros((480, 640)).astype(np.uint8)

    else:
        h, w = masks[0].shape
        semantic_map = np.zeros((h, w)).astype(np.uint8)

        for label, mask in zip(labels, masks):
            semantic_map[mask] = label

    return semantic_map
