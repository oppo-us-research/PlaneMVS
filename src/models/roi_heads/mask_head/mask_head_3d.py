"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
from torch import nn
import torch.nn.functional as F

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.poolers import Pooler


""" load our own modules """
from .loss import make_roi_mask_loss_evaluator
from .pointnet import PointNetModel


class Mask3DPooler(nn.Module):
    def __init__(self, cfg):
        super(Mask3DPooler, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD_3D.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD_3D.POOLER_SCALES

        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD_3D.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio
        )

        self.resolution = resolution
        self.pooler = pooler

    def normalize_data(self, batch_data):
        # [batch_num, 3, point_num]
        centroids = torch.mean(batch_data, dim=-1)
        batch_data = batch_data - centroids.unsqueeze(dim=-1)

        maxs = torch.max(torch.sqrt(torch.sum(batch_data ** 2, dim=1)))
        batch_data = batch_data / maxs

        return batch_data

    def forward(self, proposals, targets):
        point_maps = torch.stack([t.get_field('img_3d_points') for t in targets], dim=0).cuda()
        depth_valid_masks = torch.stack([t.get_field('depth_valid_mask') for t in targets], dim=0).cuda()
        # make it iterable to send into pooler
        point_maps = [point_maps]
        depth_valid_masks = [depth_valid_masks.unsqueeze(dim=1)]

        # [N, 3, 14, 14]
        roi_coords = self.pooler(point_maps, proposals)

        # [N, 1, 14, 14]
        roi_valid_masks = self.pooler(depth_valid_masks, proposals)

        roi_num, channels = roi_coords.size()[:2]
        roi_coords = roi_coords.view(roi_num, channels, -1)
        normalized_roi_coords = self.normalize_data(roi_coords)

        return normalized_roi_coords, roi_valid_masks

class ROIMaskHead3D(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead3D, self).__init__()
        self.cfg = cfg.clone()
        self.pooler = Mask3DPooler(cfg)
        self.predictor = PointNetModel(num_class=1, in_channels=in_channels)

        self.pooler_resolution = self.pooler.resolution
        self.resolution = cfg.MODEL.ROI_MASK_HEAD_3D.RESOLUTION

        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, proposals, gt_positive_inds, mask_targets, targets=None):
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
        roi_coords, roi_valid_masks = self.pooler(proposals, targets)
        mask_logits, feats_3d = self.predictor(roi_coords)

        mask_num = mask_logits.size(0)
        mask_logits = mask_logits.squeeze(dim=1).view(mask_num, self.pooler_resolution, self.pooler_resolution)
        mask_logits = mask_logits.unsqueeze(dim=1)

        mask_logits = F.interpolate(mask_logits, (self.resolution, self.resolution), mode='bilinear', align_corners=True).squeeze(dim=1)

        if mask_targets.numel() == 0:
            loss_mask_3d = mask_logits.sum() * 0

            return dict(loss_mask_3d=loss_mask_3d)

        roi_valid_masks = F.interpolate(roi_valid_masks, (self.resolution, self.resolution), mode='bilinear', align_corners=True).squeeze(dim=1)
        # make it to bool type
        gt_roi_valid_masks = roi_valid_masks[gt_positive_inds] > 0.5

        valid_mask_logits = mask_logits[gt_positive_inds][gt_roi_valid_masks]
        valid_mask_targets = mask_targets[gt_roi_valid_masks]

        loss_mask_3d = F.binary_cross_entropy_with_logits(
            valid_mask_logits, valid_mask_targets
        )

        return dict(loss_mask_3d=loss_mask_3d)
