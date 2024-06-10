"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch.nn import functional as F

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


""" load our own modules """
from src.models.smooth_l1_loss import smooth_l1_loss

class StereoPlaneRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        cfg,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.cfg = cfg

        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

        self.separate_pred = cfg.MODEL.SRCNN.SEPARATE_PRED

    def match_targets_to_proposals(self, proposal_left, proposal_right, target):
        left_match_quality_matrix = boxlist_iou(target, proposal_left)
        right_match_quality_matrix = boxlist_iou(target.get_field('src_bbox'), proposal_right)

        left_matched_idxs = self.proposal_matcher(left_match_quality_matrix)
        right_matched_idxs = self.proposal_matcher(right_match_quality_matrix)

        # make the intersection
        left_matched_idxs[right_matched_idxs < 0] = -1
        matched_idxs = left_matched_idxs

        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels", "src_bbox"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def separate_match_targets_to_proposals(self, proposal_left, proposal_right, target):
        left_match_quality_matrix = boxlist_iou(target, proposal_left)
        right_match_quality_matrix = boxlist_iou(target.get_field('src_bbox'), proposal_right)

        left_matched_idxs = self.proposal_matcher(left_match_quality_matrix)
        right_matched_idxs = self.proposal_matcher(right_match_quality_matrix)

        src_target = target.get_field('src_bbox')
        src_target.add_field('labels', target.get_field('src_labels'))

        assert src_target.bbox.size(0) == target.get_field('src_labels').size(0)

        target = target.copy_with_fields(['labels'])

        left_matched_targets = target[left_matched_idxs.clamp(min=0)]
        left_matched_targets.add_field('matched_idxs', left_matched_idxs)

        right_matched_targets = src_target[right_matched_idxs.clamp(min=0)]
        right_matched_targets.add_field('matched_idxs', right_matched_idxs)

        return left_matched_targets, right_matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []

        normal_cls_targets = []
        normal_res_targets = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            proposals_per_left_image, proposals_per_right_image = proposals_per_image

            matched_targets = self.match_targets_to_proposals(
                proposals_per_left_image, proposals_per_right_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            left_regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_left_image.bbox
            )

            right_regression_targets_per_image = self.box_coder.encode(
                matched_targets.get_field('src_bbox').bbox, proposals_per_left_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append((left_regression_targets_per_image, right_regression_targets_per_image))

            normal_cls_targets.append(None)
            normal_res_targets.append(None)

        return labels, regression_targets, normal_cls_targets, normal_res_targets

    def separate_prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []

        normal_cls_targets = []
        normal_res_targets = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            proposals_per_left_image, proposals_per_right_image = proposals_per_image

            left_matched_targets, right_matched_targets = self.separate_match_targets_to_proposals(
                proposals_per_left_image, proposals_per_right_image, targets_per_image
            )

            left_matched_idxs = left_matched_targets.get_field('matched_idxs')
            left_labels_per_image = left_matched_targets.get_field('labels')

            left_bg_inds = left_matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            left_labels_per_image[left_bg_inds] = 0

            left_ignore_inds = left_matched_idxs == Matcher.BETWEEN_THRESHOLDS
            left_labels_per_image[left_ignore_inds] = -1

            left_regression_targets_per_image = self.box_coder.encode(
                left_matched_targets.bbox, proposals_per_left_image.bbox
            )

            right_matched_idxs = right_matched_targets.get_field('matched_idxs')
            right_labels_per_image = right_matched_targets.get_field('labels')

            right_bg_inds = right_matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            right_labels_per_image[right_bg_inds] = 0

            right_ignore_inds = right_matched_idxs == Matcher.BETWEEN_THRESHOLDS
            right_labels_per_image[right_ignore_inds] = -1

            right_regression_targets_per_image = self.box_coder.encode(
                right_matched_targets.bbox, proposals_per_right_image.bbox
            )

            labels.append([left_labels_per_image, right_labels_per_image])
            regression_targets.append([left_regression_targets_per_image, right_regression_targets_per_image])

            normal_cls_targets.append(None)
            normal_res_targets.append(None)

        return labels, regression_targets, normal_cls_targets, normal_res_targets

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList]
            targets (list[BoxList])
        """

        if not self.separate_pred:
            labels, regression_targets, normal_cls_targets, normal_res_targets = self.prepare_targets(proposals, targets)

        else:
            labels, regression_targets, normal_cls_targets, normal_res_targets = self.separate_prepare_targets(proposals, targets)

        if not self.separate_pred:
            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        else:
            left_labels = [label[0] for label in labels]
            left_sampled_pos_inds, left_sampled_neg_inds = self.fg_bg_sampler(left_labels)

            right_labels = [label[1] for label in labels]
            right_sampled_pos_inds, right_sampled_neg_inds = self.fg_bg_sampler(right_labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image, normal_cls_targets_per_image, normal_res_targets_per_image in zip(
            labels, regression_targets, proposals, normal_cls_targets, normal_res_targets
        ):
            proposals_per_left_image, proposals_per_right_image = proposals_per_image
            regression_targets_per_left_image, regression_targets_per_right_image = regression_targets_per_image

            if not self.separate_pred:
                proposals_per_left_image.add_field("labels", labels_per_image)
            else:
                proposals_per_left_image.add_field("labels", labels_per_image[0])

            proposals_per_left_image.add_field(
                "regression_targets", regression_targets_per_left_image
            )

            if not self.separate_pred:
                proposals_per_right_image.add_field("labels", labels_per_image)
            else:
                proposals_per_right_image.add_field("labels", labels_per_image[1])

            proposals_per_right_image.add_field(
                "regression_targets", regression_targets_per_right_image
            )

        if not self.separate_pred:
            # distributed sampled proposals, that were obtained on all feature maps
            # concatenated via the fg_bg_sampler, into individual feature map levels
            for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
            ):
                img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img, as_tuple=False).squeeze(1)

                proposals_per_left_image = proposals[img_idx][0][img_sampled_inds]
                proposals[img_idx][0] = proposals_per_left_image

                proposals_per_right_image = proposals[img_idx][1][img_sampled_inds]
                proposals[img_idx][1] = proposals_per_right_image

        else:
            for img_idx, (left_pos_inds_img, left_neg_inds_img, right_pos_inds_img, right_neg_inds_img) in enumerate(
                zip(left_sampled_pos_inds, left_sampled_neg_inds, right_sampled_pos_inds, right_sampled_neg_inds)
            ):
                left_img_sampled_inds = torch.nonzero(left_pos_inds_img | left_neg_inds_img, as_tuple=False).squeeze(1)

                proposals_per_left_image = proposals[img_idx][0][left_img_sampled_inds]
                proposals[img_idx][0] = proposals_per_left_image

                right_img_sampled_inds = torch.nonzero(right_pos_inds_img | right_neg_inds_img, as_tuple=False).squeeze(1)

                proposals_per_right_image = proposals[img_idx][1][right_img_sampled_inds]
                proposals[img_idx][1] = proposals_per_right_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression, normal_cls_pred=None, normal_res_pred=None):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        if not self.separate_pred:
            class_logits = cat(class_logits, dim=0)
            box_regression = cat(box_regression, dim=0)
            device = class_logits.device

        else:
            # get the element from the list
            left_class_logits, right_class_logits = class_logits[0]
            left_box_regression, right_box_regression = box_regression[0]

            device = left_class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        if not self.separate_pred:
            # the labels are shared
            labels = cat([proposal[0].get_field("labels") for proposal in proposals], dim=0)

        else:
            left_labels = cat([proposal[0].get_field("labels") for proposal in proposals], dim=0)
            right_labels = cat([proposal[1].get_field("labels") for proposal in proposals], dim=0)

        left_regression_targets = cat(
            [proposal[0].get_field("regression_targets") for proposal in proposals], dim=0
        )

        right_regression_targets = cat(
            [proposal[1].get_field("regression_targets") for proposal in proposals], dim=0
        )

        if not self.separate_pred:
            classification_loss = F.cross_entropy(class_logits, labels)
            regression_targets = torch.cat([left_regression_targets, right_regression_targets], dim=-1)

        else:
            # concat left and right
            class_logits = torch.cat([left_class_logits, right_class_logits], dim=0)
            labels = torch.cat([left_labels, right_labels], dim=0).long()

            classification_loss = F.cross_entropy(class_logits, labels)

            box_regression = torch.cat([left_box_regression, right_box_regression], dim=0)
            regression_targets = torch.cat([left_regression_targets, right_regression_targets], dim=0)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0, as_tuple=False).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        # the labels are shared for left-right bbox
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            if not self.separate_pred:
                map_inds = 8 * labels_pos[:, None] + torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6, 7], device=device)

            else:
                map_inds = 4 * labels_pos[:, None] + torch.tensor(
                    [0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss, None, None


def make_stereo_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        0.5,
        0.1,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = StereoPlaneRCNNLossComputation(
        cfg,
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
