"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import os.path as osp

import cv2
import copy

import numpy as np

import torch
import torch.nn as nn

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
# you can ommit the "third_party.maskrcnn_main." for simplicity;
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn

""" load our own modules """
from src.models.roi_heads.roi_heads import build_roi_heads
from ..depth import build_depth


class PlaneRCNN(nn.Module):
    def __init__(self, cfg):
        super(PlaneRCNN, self).__init__()

        self.cfg = cfg

        # model initialization
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.depth_head = build_depth(cfg)

    def forward(self, images, targets=None, infos=None):
        def set_bn_eval(m):
            classname = m.__class__.__name__

            if classname.find('BatchNorm') != -1:
                m.eval()

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            # since our batch-size is usually small, 
            # we do not activate batch normalization training
            self.apply(set_bn_eval)
            return self._forward_train(images, targets)

        else:
            return self._forward_test(images, targets)

    def _forward_train(self, images, targets):
        # ref img tensors
        img_tensors = torch.cat([image['ref_img'].tensors for image in images], dim=0)

        # ref img features
        features = self.backbone(img_tensors)

        # turn into image_lists to pass into rpn
        image_lists = to_image_list(torch.cat([image['ref_img'].tensors for image in images], dim=0))

        # monocular depth
        _, attn_feats, depth_loss = self.depth_head(features, targets)
        # here actually only the image size is needed
        proposals, proposal_losses = self.rpn(image_lists, features, targets)

        # detection heads
        _, _, detector_losses = self.roi_heads(features, attn_feats, proposals, targets)

        # save losses
        losses = {}
        losses.update(proposal_losses)
        losses.update(depth_loss)
        losses.update(detector_losses)

        return losses

    def _forward_test(self, images, targets):
        images['ref_img'] = to_image_list(images['ref_img'])
        img_tensors = images['ref_img'].tensors.cuda()
        features = self.backbone(img_tensors)

        proposals, _ = self.rpn(images['ref_img'], features)

        depth, depth_feats, _ = self.depth_head(features)

        assert 'anchor_normals' in targets.fields()
        _, result, _ = self.roi_heads(features, depth_feats, proposals, targets=targets)

        assert len(result) == 1, 'Currently only support test one image at a time'
        result[0].add_field('depth', depth)

        return result
