# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# Copyright (c) 2024 OPPO. All rights reserved.
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn


""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import (
    keep_only_positive_boxes
    )

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor

""" load our own modules """
from .mask_head_3d import ROIMaskHead3D
from .mask_score_head import ROIMaskScoreHead, Mask3DPooler
from .loss import make_roi_mask_loss_evaluator
from .inference import make_roi_mask_post_processor


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_mask_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

        self.with_mask_score_head = cfg.MODEL.ROI_MASK_SCORE_HEAD.ACTIVATE
        if self.with_mask_score_head:
            self.mask_score_head = ROIMaskScoreHead(cfg, in_channels)

        self.with_mask_focal_loss = cfg.MODEL.ROI_MASK_HEAD.FOCAL_LOSS
        if self.with_mask_focal_loss:
            self.pooler_3d = Mask3DPooler(cfg)

    def forward(self, features, attn_feats, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, attn_feats, proposals)

        mask_logits, uncert_logits = self.predictor(x)

        if self.with_mask_score_head and not self.training:
            inds = torch.arange(mask_logits.size(0), device=mask_logits.device)
            label_preds = proposals[0].get_field('labels')
            mask_on_label_logits = mask_logits[inds, label_preds]

            if mask_on_label_logits.size(0) == 0:
                mask_score_logits = None

            else:
                mask_score_logits = self.mask_score_head(x, mask_on_label_logits, proposals)
                # select the scores on label
                mask_score_logits = mask_score_logits[inds, label_preds]

        if not self.training:
            if not self.with_mask_score_head:
                mask_score_logits = None

            result = self.post_processor(mask_logits, proposals, mask_score_logits)
            return x, result, {}

        if self.training and self.with_mask_focal_loss:
            roi_coords, roi_valid_masks = self.pooler_3d(proposals, targets)

        else:
            roi_coords, roi_valid_masks = None, None

        label_targets, mask_targets, plane_targets, loss_mask = self.loss_evaluator(proposals, mask_logits, targets,
                                                                                    roi_coords, roi_valid_masks,
                                                                                    uncert_logits)
        loss_dict = dict(loss_mask=loss_mask)

        if self.with_mask_score_head and self.training:
            inds = torch.arange(mask_logits.size(0), device=mask_logits.device)
            mask_on_label_logits = mask_logits[inds, label_targets]

            score_loss_dict = self.mask_score_head(x, mask_on_label_logits, proposals, targets, plane_targets, label_targets)
            loss_dict.update(score_loss_dict)

        return x, all_proposals, loss_dict


def build_roi_mask_head(cfg, in_channels):
    return ROIMaskHead(cfg, in_channels)
