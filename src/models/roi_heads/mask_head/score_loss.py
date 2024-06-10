"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
from torch.nn import functional as F


class MaskRCNNScoreLossComputation(object):
    def __init__(self):
        self.geometry_score_loss = torch.nn.MSELoss()

    def __call__(self, mask_score_logits, mask_geometry_scores):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        mask_geometry_loss = self.geometry_score_loss(mask_score_logits, mask_geometry_scores)

        return mask_geometry_loss


def make_roi_mask_score_loss_evaluator(cfg):
    loss_evaluator = MaskRCNNScoreLossComputation()

    return loss_evaluator
