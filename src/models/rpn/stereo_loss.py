"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.modeling.utils import concat_box_prediction_layers
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

""" load our own modules """
from src.models.smooth_l1_loss import smooth_l1_loss


class StereoRPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, cfg, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.cfg = cfg
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        if not self.cfg.MODEL.SRCNN.SEPARATE_PRED:
            self.copied_fields.append('src_bbox')
            self.copied_fields.append('union_bbox')

        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

        self.separate_pred = cfg.MODEL.SRCNN.SEPARATE_PRED

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        # use union bbox to assign gt to fg/bg
        match_quality_matrix = boxlist_iou(target.get_field('union_bbox'), anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def separate_match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        left_match_quality_matrix = boxlist_iou(target, anchor)
        left_matched_idxs = self.proposal_matcher(left_match_quality_matrix)

        right_match_quality_matrix = boxlist_iou(target.get_field('src_bbox'), anchor)
        right_matched_idxs = self.proposal_matcher(right_match_quality_matrix)

        src_target = target.get_field('src_bbox')

        # delete other fields
        target = target.copy_with_fields([])

        left_matched_targets = target[left_matched_idxs.clamp(min=0)]
        left_matched_targets.add_field("matched_idxs", left_matched_idxs)

        right_matched_targets = src_target[right_matched_idxs.clamp(min=0)]
        right_matched_targets.add_field("matched_idxs", right_matched_idxs)

        return left_matched_targets, right_matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # TODO: make sure the order for left_bbox, right_bbox
            # and union_bbox are consistent
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            left_regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            right_regression_targets_per_image = self.box_coder.encode(
                matched_targets.get_field('src_bbox').bbox, anchors_per_image.bbox
            )

            regression_targets_per_image = torch.cat([left_regression_targets_per_image, right_regression_targets_per_image], dim=-1)

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def separate_prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            left_matched_targets, right_matched_targets = self.separate_match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )
            left_matched_idxs = left_matched_targets.get_field('matched_idxs')
            left_labels_per_image = self.generate_labels_func(left_matched_targets)
            left_labels_per_image = left_labels_per_image.to(dtype=torch.float32)

            left_bg_indices = left_matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            left_labels_per_image[left_bg_indices] = 0

            if "not_visibility" in self.discard_cases:
                left_labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            if "between_thresholds" in self.discard_cases:
                left_inds_to_discard = left_matched_idxs == Matcher.BETWEEN_THRESHOLDS
                left_labels_per_image[left_inds_to_discard] = -1

            left_regression_targets_per_image = self.box_coder.encode(
                left_matched_targets.bbox, anchors_per_image.bbox
            )

            right_matched_idxs = right_matched_targets.get_field('matched_idxs')
            right_labels_per_image = self.generate_labels_func(right_matched_targets)
            right_labels_per_image = right_labels_per_image.to(dtype=torch.float32)

            right_bg_indices = right_matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            right_labels_per_image[right_bg_indices] = 0

            if "not_visibility" in self.discard_cases:
                right_labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            if "between_thresholds" in self.discard_cases:
                right_inds_to_discard = right_matched_idxs == Matcher.BETWEEN_THRESHOLDS
                right_labels_per_image[right_inds_to_discard] = -1

            right_regression_targets_per_image = self.box_coder.encode(
                right_matched_targets.bbox, anchors_per_image.bbox
            )

            labels_per_image = torch.stack([left_labels_per_image, right_labels_per_image], dim=-1)
            regression_targets_per_image = torch.cat([left_regression_targets_per_image, right_regression_targets_per_image], dim=-1)

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[list[BoxList]])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]

        if self.separate_pred:
            labels, regression_targets = self.separate_prepare_targets(anchors, targets)
            left_labels = [label[:, 0] for label in labels]
            left_sampled_pos_inds, left_sampled_neg_inds = self.fg_bg_sampler(left_labels)
            left_sampled_pos_inds = torch.nonzero(torch.cat(left_sampled_pos_inds, dim=0)).squeeze(1)
            left_sampled_neg_inds = torch.nonzero(torch.cat(left_sampled_neg_inds, dim=0)).squeeze(1)

            left_sampled_inds = torch.cat([left_sampled_pos_inds, left_sampled_neg_inds], dim=0)
            objectness, box_regression = \
                concat_box_prediction_layers(objectness, box_regression, reg_dim=8)

            left_objectness = objectness[:, 0].squeeze()

            right_labels = [label[:, 1] for label in labels]
            right_sampled_pos_inds, right_sampled_neg_inds = self.fg_bg_sampler(right_labels)
            right_sampled_pos_inds = torch.nonzero(torch.cat(right_sampled_pos_inds, dim=0)).squeeze(1)
            right_sampled_neg_inds = torch.nonzero(torch.cat(right_sampled_neg_inds, dim=0)).squeeze(1)

            right_sampled_inds = torch.cat([right_sampled_pos_inds, right_sampled_neg_inds], dim=0)
            right_objectness = objectness[:, 1].squeeze()

            labels = torch.cat(labels, dim=0)
            regression_targets = torch.cat(regression_targets, dim=0)

            box_regression = torch.cat([box_regression[left_sampled_pos_inds, :4], box_regression[right_sampled_pos_inds, 4:]], dim=0)
            regression_targets = torch.cat([regression_targets[left_sampled_pos_inds, :4], regression_targets[right_sampled_pos_inds, 4:]], dim=0)

            box_loss = smooth_l1_loss(
                box_regression,
                regression_targets,
                beta=1.0 / 9,
                size_average=False,
            ) / (left_sampled_inds.numel() + right_sampled_inds.numel())

            objectness = torch.cat([left_objectness[left_sampled_inds], right_objectness[right_sampled_inds]], dim=0)
            labels = torch.cat([labels[:, 0][left_sampled_inds], labels[:, 1][right_sampled_inds]], dim=0)

            objectness_loss = F.binary_cross_entropy_with_logits(
                objectness, labels
            )

        else:
            labels, regression_targets = self.prepare_targets(anchors, targets)

            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
            sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
            sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

            objectness, box_regression = \
                    concat_box_prediction_layers(objectness, box_regression, reg_dim=8)

            objectness = objectness.squeeze()

            labels = torch.cat(labels, dim=0)
            regression_targets = torch.cat(regression_targets, dim=0)

            box_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1.0 / 9,
                size_average=False,
            ) / (sampled_inds.numel())

            objectness_loss = F.binary_cross_entropy_with_logits(
                objectness[sampled_inds], labels[sampled_inds]
            )

        return objectness_loss, box_loss


# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_stereo_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = StereoRPNLossComputation(
        cfg,
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels
    )
    return loss_evaluator
