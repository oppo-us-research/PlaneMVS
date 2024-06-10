"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch import nn

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor


""" load our own modules """
from .normal_predictors import make_normal_predictor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator


class PlaneROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(PlaneROIBoxHead, self).__init__()
        self.cfg = cfg

        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        
        # ------------------------------
        # Added by PlaneMVS's authors;
        # ------------------------------
        if self.cfg.MODEL.METHOD == 'single' or self.cfg.MODEL.METHOD == 'refine':
            self.normal_predictor = make_normal_predictor(cfg, self.feature_extractor.out_channels)
        self.wo_normal_head = cfg.MODEL.ROI_BOX_HEAD.WO_NORMAL_HEAD
    

    # ------------------------------
    # Updated by PlaneMVS's authors;
    # ------------------------------
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

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        flattened_x, class_logits, box_regression, class_uncert_logits, box_uncert_logits = self.predictor(x)

        if not self.wo_normal_head and (self.cfg.MODEL.METHOD == 'single' or self.cfg.MODEL.METHOD == 'refine'):
            normal_cls_pred, normal_res_pred = self.normal_predictor(x)

            if not self.training:
                assert anchor_normals is not None
                result = self.post_processor((class_logits, box_regression), proposals, (normal_cls_pred, normal_res_pred), anchor_normals)

                return x, result, {}

            loss_classifier, loss_box_reg, loss_normal_cls, loss_normal_res = self.loss_evaluator(
                [class_logits], [box_regression], [normal_cls_pred], [normal_res_pred]
            )

            loss_dict = dict(
                loss_classifier=loss_classifier,
                loss_box_reg=loss_box_reg,
                loss_normal_cls=loss_normal_cls,
                loss_normal_res=loss_normal_res
            )

        elif self.wo_normal_head or self.cfg.MODEL.METHOD == 'stereo' or self.cfg.MODEL.METHOD == 'consistency':
            if not self.training:
                result = self.post_processor((class_logits, box_regression), proposals)

                return x, result, {}

            loss_classifier, loss_box_reg, _, _ = self.loss_evaluator(
                [class_logits], [box_regression], None, None, class_uncert_logits, box_uncert_logits
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


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return PlaneROIBoxHead(cfg, in_channels)
