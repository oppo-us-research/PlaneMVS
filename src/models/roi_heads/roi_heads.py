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

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
# you can aslo ommit the "third_party.maskrcnn_main." for simplicity;
from maskrcnn_benchmark.modeling.roi_heads.keypoint_head.keypoint_head import build_roi_keypoint_head


""" load our own modules """
from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .box_head.stereo_box_head import build_stereo_roi_box_head
from .mask_head.stereo_mask_head import build_stereo_roi_mask_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, attn_feats, proposals, targets=None, bbox_no_grad=False):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption

        if not self.training and (self.cfg.MODEL.METHOD == 'single' or self.cfg.MODEL.METHOD == 'refine'):
            anchor_normals = targets.get_field('anchor_normals')
        else:
            anchor_normals = None

        # if we need predicted mask with gradient, we can freeze the bbox head to save memory
        if bbox_no_grad:
            with torch.no_grad():
                x, detections, loss_box = self.box(features, proposals, targets, anchor_normals=anchor_normals)

        else:
            x, detections, loss_box = self.box(features, proposals, targets, anchor_normals=anchor_normals)

        losses.update(loss_box)

        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, attn_feats, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)

        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        if cfg.MODEL.METHOD == 'srcnn':
            roi_heads.append(("box", build_stereo_roi_box_head(cfg, in_channels)))
        else:
            roi_heads.append(("box", build_roi_box_head(cfg, in_channels))) # go here

    if cfg.MODEL.MASK_ON:
        if cfg.MODEL.METHOD == 'srcnn':
            roi_heads.append(("mask", build_stereo_roi_mask_head(cfg, in_channels)))
        else:
            roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels))) # go here
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
