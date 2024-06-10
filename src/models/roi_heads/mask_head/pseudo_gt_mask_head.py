"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
from torch import nn
from torch.nn import functional as F

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.roi_heads.mask_head.loss import project_masks_on_boxes



class PseudoMaskLossComputation(object):
    def __init__(self, cfg):
        self.discretization_size = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION

    def __call__(self, proposals, mask_logits, targets):
        labels = [proposal.get_field('labels') for proposal in proposals]
        labels = cat(labels, dim=0)

        mask_targets = []
        valid_masks = []

        for proposal, target in zip(proposals, targets):
            segmentation_masks = target.get_field('pseudo_masks')

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, proposal, self.discretization_size
            )
            mask_targets.append(masks_per_image)

            valid_mask = target.get_field('valid_mask')
            valid_mask_per_image = project_masks_on_boxes(
                valid_mask, proposal, self.discretization_size
            )

            valid_masks.append(valid_mask_per_image)

        mask_targets = cat(mask_targets, dim=0)
        valid_masks = cat(valid_masks, dim=0)

        # turn into bool
        valid_masks = valid_masks > 0.5

        mask_logits = mask_logits[torch.arange(mask_logits.size(0)), labels, ...]

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[valid_masks], mask_targets[valid_masks]
        )

        return mask_loss


class PseudoROIMaskHead(torch.nn.Module):
    def __init__(self, cfg):
        super(PseudoROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.loss_evaluator = PseudoMaskLossComputation(cfg)

    def forward(self, mask_logits, proposals, pseudo_targets):
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
        mask_logits = cat(mask_logits, dim=0)
        loss_mask = self.loss_evaluator(proposals, mask_logits, pseudo_targets)

        return loss_mask


def build_roi_mask_head(cfg, in_channels):
    return PseudoROIMaskHead(cfg, in_channels)
