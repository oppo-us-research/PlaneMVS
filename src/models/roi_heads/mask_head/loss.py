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
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.roi_heads.mask_head.loss import project_masks_on_boxes

""" load our own modules """
from src.models.smooth_l1_loss import smooth_l1_loss


class MaskRCNNLossComputation(object):
    def __init__(self, cfg, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        # ------------------------------
        # Added by PlaneMVS's authors;
        # ------------------------------
        self.with_mask_score_head = cfg.MODEL.ROI_MASK_SCORE_HEAD.ACTIVATE
        self.with_mask_focal_loss = cfg.MODEL.ROI_MASK_HEAD.FOCAL_LOSS

    # ------------------------------
    # Added by PlaneMVS's authors;
    # ------------------------------
    def compute_plane_weights(self, mask_logits, mask_targets, plane_targets, roi_coords, dist_scale=0.25):
        assert mask_logits.size() == mask_targets.size()
        bin_masks = torch.sigmoid(mask_logits) > 0.5

        roi_coords = roi_coords.view(roi_coords.size(0), 3, -1)
        bin_masks = bin_masks.squeeze(dim=1)

        dists = []
        focal_weights = []

        h, w = bin_masks.size()[-2:]

        for bin_mask, mask_target, plane, coords in zip(bin_masks, mask_targets, plane_targets, roi_coords):
            dist = torch.abs(plane.unsqueeze(dim=0) @ coords + \
                             torch.ones((1, coords.size(-1)), device=coords.device))
            dist = dist.squeeze(dim=0).view(h, w)

            weight = torch.ones(mask_target.size(), device=bin_mask.device)
            fn_mask = ((~bin_mask) * mask_target) > 0.5
            weight[fn_mask] = torch.exp(dist_scale * (1 - dist)).clamp(min=1)[fn_mask]

            fp_mask = (bin_mask * (1 - mask_target)) > 0.5
            weight[fp_mask] = torch.exp(dist_scale * dist)[fp_mask]

            focal_weights.append(weight)

        focal_weights = torch.stack(focal_weights, dim=0)

        return focal_weights

    # ------------------------------
    # Updated by PlaneMVS's authors;
    # ------------------------------
    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)

        if self.with_mask_score_head or self.with_mask_focal_loss:
            target = target.copy_with_fields(["labels", "masks", "plane_instances"])
        else:
            # Mask RCNN needs "labels" and "masks "fields for creating the targets
            target = target.copy_with_fields(["labels", "masks"])

        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    # ------------------------------
    # Updated by PlaneMVS's authors;
    # ------------------------------
    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        planes = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0, as_tuple=False).squeeze(1)

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.discretization_size
            )

            labels.append(labels_per_image)
            masks.append(masks_per_image)

            if self.with_mask_score_head or self.with_mask_focal_loss:
                planes_per_image = matched_targets.get_field('plane_instances')
                positive_planes_per_image = planes_per_image[positive_inds]

                planes.append(planes_per_image)

        return labels, masks, planes

    # ------------------------------
    # Updated by PlaneMVS's authors;
    # ------------------------------
    def __call__(self, proposals, mask_logits, targets, roi_coords=None, roi_valid_masks=None, uncert_logits=None):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, mask_targets, plane_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        if self.with_mask_score_head or self.with_mask_focal_loss:
            plane_targets = cat(plane_targets, dim=0)

        else:
            plane_targets = None

        positive_inds = torch.nonzero(labels > 0, as_tuple=False).squeeze(1)
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return labels_pos, mask_targets, plane_targets, mask_logits.sum() * 0

        if self.with_mask_focal_loss:
            pos_mask_logits = mask_logits[positive_inds, labels_pos]
            focal_weights = self.compute_plane_weights(pos_mask_logits, mask_targets,
                                                        plane_targets, roi_coords)

            # turn into bool
            roi_valid_masks = (roi_valid_masks > 0.5).squeeze(dim=1)

            valid_mask_logits = pos_mask_logits[roi_valid_masks]
            valid_mask_targets = mask_targets[roi_valid_masks]

            focal_weights = focal_weights[roi_valid_masks]

            mask_loss = F.binary_cross_entropy_with_logits(
                valid_mask_logits, valid_mask_targets,
                weight=focal_weights
            )

        elif uncert_logits is not None:
            eps = 1e-12

            conf = torch.sigmoid(uncert_logits).squeeze(dim=1)
            mask_pred_on_label = torch.sigmoid(mask_logits[positive_inds, labels_pos]).squeeze(dim=1)

            conf = torch.clamp(conf, 0. + eps, 1. - eps)
            mask_pred_on_label = torch.clamp(mask_pred_on_label, 0. + eps, 1. - eps)

            mask_pred_new = mask_pred_on_label * conf + mask_targets * (1 - conf)

            mask_loss = F.binary_cross_entropy(
                mask_pred_new, mask_targets)

            confidence_loss = -torch.log(conf).mean()

            mask_loss = mask_loss + confidence_loss

        else:
            mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits[positive_inds, labels_pos], mask_targets
            )

        return labels_pos, mask_targets, plane_targets, mask_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        cfg, matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
    )

    return loss_evaluator
