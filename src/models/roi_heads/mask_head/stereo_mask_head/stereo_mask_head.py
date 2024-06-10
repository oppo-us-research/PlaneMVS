# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .stereo_roi_mask_feature_extractors import make_stereo_roi_mask_feature_extractor
from .stereo_roi_mask_predictors import make_stereo_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .stereo_loss import make_stereo_roi_mask_loss_evaluator


def keep_only_positive_boxes(boxes, separate_pred=False):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], (list, tuple))
    assert isinstance(boxes[0][0], BoxList)
    assert boxes[0][0].has_field("labels")

    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_left_image, boxes_per_right_image in boxes:
        if not separate_pred:
            labels = boxes_per_left_image.get_field("labels")
            inds_mask = labels > 0
            inds = inds_mask.nonzero().squeeze(1)
            positive_boxes.append((boxes_per_left_image[inds], boxes_per_right_image[inds]))
            positive_inds.append(inds_mask)

        else:
            left_labels = boxes_per_left_image.get_field('labels')
            left_inds_mask = left_labels > 0
            left_inds = left_inds_mask.nonzero().squeeze(1)

            right_labels = boxes_per_right_image.get_field('labels')
            right_inds_mask = right_labels > 0
            right_inds = right_inds_mask.nonzero().squeeze(1)

            positive_boxes.append([boxes_per_left_image[left_inds], boxes_per_right_image[right_inds]])
            positive_inds.append([left_inds_mask, right_inds_mask])

    return positive_boxes, positive_inds


class StereoROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(StereoROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_stereo_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor = make_stereo_roi_mask_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_stereo_roi_mask_loss_evaluator(cfg)

        self.get_soft_mask = cfg.MODEL.STEREO.WITH_PIXEL_INSTANCE_CONSISTENCY_LOSS

        self.separate_pred = cfg.MODEL.SRCNN.SEPARATE_PRED

    def forward(self, features, proposals, targets=None):
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
            proposals, positive_inds = keep_only_positive_boxes(proposals, self.separate_pred)
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            raise NotImplementedError

        else:
            x = self.feature_extractor(features, proposals)

        mask_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)

        return x, all_proposals, dict(loss_mask=loss_mask)


def build_stereo_roi_mask_head(cfg, in_channels):
    return StereoROIMaskHead(cfg, in_channels)
