"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""
import torch
from torch import nn

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.poolers import Pooler

""" load our own modules """
from .roi_mask_score_feature_extractors import make_roi_mask_score_feature_extractor
from .roi_mask_score_predictors import make_roi_mask_score_predictor
from .score_loss import make_roi_mask_score_loss_evaluator


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

    def forward(self, proposals, targets):
        point_maps = torch.stack([t.get_field('img_3d_points') for t in targets], dim=0).cuda()
        depth_valid_masks = torch.stack([t.get_field('depth_valid_mask') for t in targets], dim=0).cuda()

        point_maps = [point_maps]
        depth_valid_masks = [depth_valid_masks.unsqueeze(dim=1)]

        roi_coords = self.pooler(point_maps, proposals)
        roi_valid_masks = self.pooler(depth_valid_masks, proposals)

        return roi_coords, roi_valid_masks


class ROIMaskScoreHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskScoreHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_score_feature_extractor(cfg, in_channels + 1)

        # plus prediction dimension
        self.predictor = make_roi_mask_score_predictor(
            cfg, self.feature_extractor.out_channels)

        self.pooler_3d = Mask3DPooler(cfg)
        self.loss_evaluator = make_roi_mask_score_loss_evaluator(cfg)

    # since the mask_logits have been concated, we compute the score targets here
    # to avoid being into the for-loop
    def make_score_targets(self, mask_logits, plane_targets, roi_coords, roi_valid_masks, inlier_thresh=0.03):
        bin_masks = torch.sigmoid(mask_logits) > 0.5
        geometry_scores = []

        # [batch_num, 3, point_num]
        roi_coords = roi_coords.view(roi_coords.size(0), 3, -1)

        # make binary
        roi_valid_masks = roi_valid_masks.squeeze(dim=1).view(roi_valid_masks.size(0), -1)
        roi_valid_masks = roi_valid_masks > 0.5

        bin_masks = bin_masks.squeeze(dim=1).view(bin_masks.size(0), -1)
        valid_idxs = []

        for idx, (bin_mask, plane, coords, coord_valid_mask) in enumerate(zip(bin_masks, plane_targets, roi_coords, roi_valid_masks)):
            valid_mask = bin_mask * coord_valid_mask

            if valid_mask.sum() == 0:
                continue

            valid_points = coords[:, valid_mask]

            dist = torch.mean(torch.abs(plane.unsqueeze(dim=0) @ valid_points + \
                                        torch.ones((1, valid_points.size(-1)), device=valid_points.device)))

            geometry_score = torch.exp(inlier_thresh - 5 * dist).clamp(max=1)
            geometry_scores.append(geometry_score)

            valid_idxs.append(idx)

        geometry_scores = torch.stack(geometry_scores, dim=0)
        valid_idxs = torch.tensor(valid_idxs).long().to(geometry_scores.device)

        return geometry_scores, valid_idxs

    def forward(self, mask_feats, mask_logits_on_label, proposals, targets=None, plane_targets=None, label_targets=None):
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
        mask_geometry_feats = self.feature_extractor(mask_feats, mask_logits_on_label.unsqueeze(dim=1))
        mask_geometry_logits = self.predictor(mask_geometry_feats)

        if not self.training:
            return mask_geometry_logits.clamp(max=1)

        else:
            assert label_targets is not None

            roi_coords, roi_valid_masks = self.pooler_3d(proposals, targets)
            geometry_score_targets, valid_idxs = self.make_score_targets(mask_logits_on_label, plane_targets, roi_coords, roi_valid_masks)

            inds = torch.arange(mask_geometry_logits.size(0), device=mask_geometry_logits.device)
            mask_geometry_logits_on_label = mask_geometry_logits[inds, label_targets]

            # only keep the logits with valid targets
            mask_geometry_logits_on_label = mask_geometry_logits_on_label[valid_idxs]

            loss_geometry_score = self.loss_evaluator(mask_geometry_logits_on_label, geometry_score_targets)
            loss_dict = dict(loss_geometry_score=loss_geometry_score)

            return loss_dict
