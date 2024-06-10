"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
from torch import nn

from .stereo_roi_box_feature_extractors import make_stereo_roi_box_feature_extractor
from .stereo_roi_box_predictor import make_stereo_roi_box_predictor
from .inference import make_roi_box_post_processor
from .stereo_loss import make_stereo_roi_box_loss_evaluator


class StereoPlaneROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(StereoPlaneROIBoxHead, self).__init__()
        self.cfg = cfg

        self.feature_extractor = make_stereo_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_stereo_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)

        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_stereo_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None, anchor_normals=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # NOTE: here proposal are a left-right pair, the scores are shared
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        flattened_x, class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)

            return x, result, {}

        loss_classifier, loss_box_reg, _, _ = self.loss_evaluator(
            [class_logits], [box_regression]
        )

        loss_dict = dict(
            loss_classifier=loss_classifier,
            loss_box_reg=loss_box_reg
        )

        return (
            x,
            proposals,
            loss_dict,
        )


def build_stereo_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return StereoPlaneROIBoxHead(cfg, in_channels)
