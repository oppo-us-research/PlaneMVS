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
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat

""" load our own modules """
from src.models.smooth_l1_loss import smooth_l1_loss


class PlaneRCNNLossComputation(object):
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

        self.wo_normal_head = cfg.MODEL.ROI_BOX_HEAD.WO_NORMAL_HEAD

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets

        if self.cfg.MODEL.METHOD == 'single' or self.cfg.MODEL.METHOD == 'refine':
            target = target.copy_with_fields(["labels", "normal_cls", "normal_res"])
        else:
            target = target.copy_with_fields(["labels"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []

        normal_cls_targets = []
        normal_res_targets = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
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
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

            if not self.wo_normal_head and (self.cfg.MODEL.METHOD == 'single' or self.cfg.MODEL.METHOD == 'refine'):
                normal_cls_targets.append(matched_targets.get_field('normal_cls'))
                normal_res_targets.append(matched_targets.get_field('normal_res'))

            else:
                normal_cls_targets.append(None)
                normal_res_targets.append(None)

        return labels, regression_targets, normal_cls_targets, normal_res_targets

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets, normal_cls_targets, normal_res_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image, normal_cls_targets_per_image, normal_res_targets_per_image in zip(
            labels, regression_targets, proposals, normal_cls_targets, normal_res_targets
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

            if not self.wo_normal_head and (self.cfg.MODEL.METHOD == 'single' or self.cfg.MODEL.METHOD == 'refine'):
                proposals_per_image.add_field('normal_cls', normal_cls_targets_per_image)
                proposals_per_image.add_field('normal_res', normal_res_targets_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img, as_tuple=False).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def encode_onehot(self, labels, n_classes):
        onehot = torch.FloatTensor(labels.size()[0], n_classes)
        labels = labels.data
        if labels.is_cuda:
            onehot = onehot.cuda()
        onehot.zero_()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        onehot = onehot.long()

        return onehot

    def __call__(self, class_logits, box_regression, normal_cls_pred=None, normal_res_pred=None, class_uncert_logits=None, box_uncert_logits=None):
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

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not self.wo_normal_head and (self.cfg.MODEL.METHOD == 'single' or self.cfg.MODEL.METHOD == 'refine'):
            normal_cls_pred = cat(normal_cls_pred, dim=0)
            normal_res_pred = cat(normal_res_pred, dim=0)

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        if not self.wo_normal_head and (self.cfg.MODEL.METHOD == 'single' or self.cfg.MODEL.METHOD == 'refine'):
            normal_cls_targets = cat([proposal.get_field('normal_cls') for proposal in proposals], dim=0).long()
            normal_res_targets = cat([proposal.get_field('normal_res') for proposal in proposals], dim=0)

        if class_uncert_logits is not None:
            conf = F.softmax(class_uncert_logits, dim=1)
            labels_onehot = self.encode_onehot(labels, conf.size(-1))

            class_pred = F.softmax(class_logits, dim=1)

            eps = 1e-12
            class_pred = torch.clamp(class_pred, 0. + eps, 1. - eps)
            conf = torch.clamp(conf, 0. + eps, 1. - eps)

            new_class_pred = class_pred * conf + labels_onehot * (1 - conf)

            # origin_classification_loss = F.nll_loss(torch.log(class_pred), labels)

            # nll_loss accept log likelihood as input
            classification_loss = F.nll_loss(torch.log(new_class_pred), labels)
            confidence_loss = torch.mean(-torch.log(conf))

            classification_loss = 0.5 * (classification_loss + confidence_loss)

        else:
            classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0, as_tuple=False).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        if box_uncert_logits is not None:
            box_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds_subset],
                size_average=False,
                beta=1,
                reduction='none'
            )
            box_loss = box_loss * torch.exp(-box_uncert_logits[sampled_pos_inds_subset].squeeze(dim=-1)) + 0.5 * box_uncert_logits[sampled_pos_inds_subset].squeeze(dim=-1)
            box_loss = box_loss.sum() / labels.numel()

        else:
            box_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds_subset],
                size_average=False,
                beta=1,
            )
            box_loss = box_loss / labels.numel()

        if not self.wo_normal_head and (self.cfg.MODEL.METHOD == 'single' or self.cfg.MODEL.METHOD == 'refine'):
            normal_cls_loss = F.cross_entropy(
                normal_cls_pred[sampled_pos_inds_subset],
                normal_cls_targets[sampled_pos_inds_subset]
            )

            normal_cls_pos = normal_cls_targets[sampled_pos_inds_subset]
            map_inds = 3 * normal_cls_pos[:, None] + torch.tensor([0, 1, 2], device=device)

            normal_res_loss = smooth_l1_loss(
                normal_res_pred[sampled_pos_inds_subset[:, None], map_inds],
                normal_res_targets[sampled_pos_inds_subset],
                size_average=False,
                beta=1
            )

            normal_res_loss = normal_res_loss / labels.numel()

            return classification_loss, box_loss, normal_cls_loss, normal_res_loss

        else:
            return classification_loss, box_loss, None, None


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = PlaneRCNNLossComputation(
        cfg,
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
