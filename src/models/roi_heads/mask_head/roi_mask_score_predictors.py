"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

from torch import nn
from torch.nn import functional as F


class MaskScorePredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskScorePredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.mask_score = nn.Linear(in_channels, num_classes)

        nn.init.normal_(self.mask_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.mask_score.bias, 0)

    def forward(self, x):
        mask_score = self.mask_score(x)

        return mask_score


def make_roi_mask_score_predictor(cfg, in_channels):
    return MaskScorePredictor(cfg, in_channels)
