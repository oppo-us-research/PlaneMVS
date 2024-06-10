# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    # FIXME: CPU computation bottleneck, this should be parallelized
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation.
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class StereoMaskRCNNLossComputation(object):
    def __init__(self, cfg, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size

        self.separate_pred = cfg.MODEL.SRCNN.SEPARATE_PRED

    def match_targets_to_proposals(self, proposal_left, proposal_right, target):
        left_match_quality_matrix = boxlist_iou(target, proposal_left)
        right_match_quality_matrix = boxlist_iou(target.get_field('src_bbox'), proposal_right)

        left_matched_idxs = self.proposal_matcher(left_match_quality_matrix)
        right_matched_idxs = self.proposal_matcher(right_match_quality_matrix)

        left_matched_idxs[right_matched_idxs < 0] = -1
        matched_idxs = left_matched_idxs

        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks", "src_masks"])
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
        src_target.add_field('masks', target.get_field('src_masks'))
        src_target.add_field('labels', target.get_field('src_labels'))

        target = target.copy_with_fields(['labels', 'masks'])

        left_matched_targets = target[left_matched_idxs.clamp(min=0)]
        left_matched_targets.add_field('matched_idxs', left_matched_idxs)

        right_matched_targets = src_target[right_matched_idxs.clamp(min=0)]
        right_matched_targets.add_field('matched_idxs', right_matched_idxs)

        return left_matched_targets, right_matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            proposals_per_left_image, proposals_per_right_image = proposals_per_image

            matched_targets = self.match_targets_to_proposals(
                proposals_per_left_image, proposals_per_right_image, targets_per_image
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

            left_segmentation_masks = matched_targets.get_field("masks")
            left_segmentation_masks = left_segmentation_masks[positive_inds]

            left_positive_proposals = proposals_per_left_image[positive_inds]

            masks_per_left_image = project_masks_on_boxes(
                left_segmentation_masks, left_positive_proposals, self.discretization_size
            )

            right_segmentation_masks = matched_targets.get_field("src_masks")
            right_segmentation_masks = right_segmentation_masks[positive_inds]

            right_positive_proposals = proposals_per_right_image[positive_inds]

            masks_per_right_image = project_masks_on_boxes(
                right_segmentation_masks, right_positive_proposals, self.discretization_size
            )

            labels.append(labels_per_image)
            masks.append((masks_per_left_image, masks_per_right_image))

        return labels, masks

    def separate_prepare_targets(self, proposals, targets):
        labels = []
        masks = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            proposals_per_left_image, proposals_per_right_image = proposals_per_image

            left_matched_targets, right_matched_targets = self.separate_match_targets_to_proposals(
                proposals_per_left_image, proposals_per_right_image, targets_per_image
            )
            left_matched_idxs = left_matched_targets.get_field('matched_idxs')

            left_labels_per_image = left_matched_targets.get_field('labels')
            left_labels_per_image = left_labels_per_image.to(dtype=torch.int64)

            left_neg_inds = left_matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            left_labels_per_image[left_neg_inds] = 0

            left_positive_inds = torch.nonzero(left_labels_per_image > 0, as_tuple=False).squeeze(1)

            left_segmentation_masks = left_matched_targets.get_field('masks')
            left_segmentation_masks = left_segmentation_masks[left_positive_inds]

            left_positive_proposals = proposals_per_left_image[left_positive_inds]

            masks_per_left_image = project_masks_on_boxes(
                left_segmentation_masks, left_positive_proposals, self.discretization_size
            )

            right_matched_idxs = right_matched_targets.get_field('matched_idxs')

            right_labels_per_image = right_matched_targets.get_field('labels')
            right_labels_per_image = right_labels_per_image.to(dtype=torch.int64)

            right_neg_inds = right_matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            right_labels_per_image[right_neg_inds] = 0

            right_positive_inds = torch.nonzero(right_labels_per_image > 0, as_tuple=False).squeeze(1)

            right_segmentation_masks = right_matched_targets.get_field('masks')
            right_segmentation_masks = right_segmentation_masks[right_positive_inds]

            right_positive_proposals = proposals_per_right_image[right_positive_inds]

            masks_per_right_image = project_masks_on_boxes(
                right_segmentation_masks, right_positive_proposals, self.discretization_size
            )

            labels.append([left_labels_per_image, right_labels_per_image])
            masks.append([masks_per_left_image, masks_per_right_image])

        return labels, masks

    def __call__(self, proposals, mask_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        if not self.separate_pred:
            labels, mask_targets = self.prepare_targets(proposals, targets)

        else:
            labels, mask_targets = self.separate_prepare_targets(proposals, targets)

        if not self.separate_pred:
            labels = cat(labels, dim=0)

            # in this case the left and right labels are the same
            labels = torch.cat([labels, labels], dim=0)

        else:
            left_labels = cat([label[0] for label in labels], dim=0)
            right_labels = cat([label[1] for label in labels], dim=0)

            labels = torch.cat([left_labels, right_labels], dim=0)

        left_mask_targets = [mask_target[0] for mask_target in mask_targets]
        right_mask_targets = [mask_target[1] for mask_target in mask_targets]

        left_mask_targets = cat(left_mask_targets, dim=0)
        right_mask_targets = cat(right_mask_targets, dim=0)

        mask_targets = torch.cat([left_mask_targets, right_mask_targets], dim=0)

        positive_inds = torch.nonzero(labels > 0, as_tuple=False).squeeze(1)
        labels_pos = labels[positive_inds]

        mask_logits = torch.cat(mask_logits, dim=0)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[positive_inds, labels_pos], mask_targets
        )

        return mask_loss


def make_stereo_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        0.5,
        0.1,
        allow_low_quality_matches=False,
    )

    loss_evaluator = StereoMaskRCNNLossComputation(
        cfg, matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
    )

    return loss_evaluator
