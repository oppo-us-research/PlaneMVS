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

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
# you can aslo ommit the "third_party.maskrcnn_main." for simplicity;
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn


""" load our own modules """
from src.models.roi_heads.roi_heads import build_roi_heads
from ..depth import build_depth
from ..stereo import build_depth_stereo


class PlaneDepthStereo(nn.Module):
    def __init__(self, cfg):
        super(PlaneDepthStereo, self).__init__()

        self.cfg = cfg

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.wo_depth_head = cfg.MODEL.STEREO.WO_SINGLE_DEPTH

        if not self.wo_depth_head:
            self.depth_head = build_depth(cfg)

        self.stereo_net = build_depth_stereo(cfg)

        self.wo_det_loss = cfg.MODEL.STEREO.WO_DET_LOSS

        self.with_loss_term_uncertainty = cfg.MODEL.STEREO.WITH_LOSS_TERM_UNCERTAINTY
        self.loss_term_num = cfg.MODEL.STEREO.LOSS_TERM_NUM

        if self.with_loss_term_uncertainty:
            self.loss_term_uncert = nn.Parameter(torch.rand(self.loss_term_num).to('cuda'), requires_grad=True)
            torch.nn.init.constant_(self.loss_term_uncert, -1.0)

    def forward(self, images, targets=None, infos=None):
        def set_bn_eval(m):
            classname = m.__class__.__name__

            if classname.find('BatchNorm') != -1:
                m.eval()

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            if not self.cfg.MODEL.ACTIVATE_BN_TRAIN:
                self.apply(set_bn_eval)

            return self._forward_train(images, targets, infos)

        else:
            return self._forward_test(images, targets)

    def _forward_train(self, images, targets, infos=None):
        ref_img_tensors = torch.cat([image['ref_img'].tensors for image in images], dim=0)
        src_img_tensors = torch.cat([image['src_img'].tensors for image in images], dim=0)

        ref_features = self.backbone(ref_img_tensors)
        src_features = self.backbone(src_img_tensors)

        # get the coarse layer feature from fpn levels,
        # which should be 1/4 H * 1/4 W resolution
        ref_bot_feat = ref_features[0]
        src_bot_feat = src_features[0]

        hypos = torch.stack([image['depth_hypos'] for image in images], dim=0)

        pred_depth_map, refined_depth_map, stereo_losses = self.stereo_net(ref_bot_feat, src_bot_feat, hypos, ref_img_tensors, targets)

        image_lists = to_image_list(ref_img_tensors)

        attn_feats = None
        # here actually only the image size is needed
        proposals, proposal_losses = self.rpn(image_lists, ref_features, targets)
        _, _, detector_losses = self.roi_heads(ref_features, attn_feats, proposals, targets)

        losses = OrderedDict()

        if not self.wo_depth_head:
            losses.update(depth_loss)

        if not self.wo_det_loss:
            losses.update(proposal_losses)
            losses.update(detector_losses)

        losses.update(stereo_losses)

        if self.with_loss_term_uncertainty:
            for idx, (key, val) in enumerate(losses.items()):
                losses[key] = losses[key] * torch.exp(-self.loss_term_uncert[idx]) + self.loss_term_uncert[idx]

        return losses

    def make_camera_grid(self, targets, h, w):
        intrinsic = targets.get_field('intrinsic')[:3, :3]
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

        grid = torch.ones(3, h, w).type_as(intrinsic)
        grid[0, ...] = xs
        grid[1, ...] = ys

        camera_grid = intrinsic.inverse() @ grid.contiguous().view(3, -1).type_as(intrinsic)
        camera_grid = camera_grid.view(3, h, w)

        return camera_grid

    def fit_planes(self, result, targets):
        masks = result[0].get_field('mask').squeeze(dim=1)
        depth = result[0].get_field('depth')

        if masks.shape[0] == 0:
            return depth

        h, w = depth.size()[-2:]
        camera_grid = self.make_camera_grid(targets, h, w).type_as(depth)

        points = camera_grid * depth

        points = points.cpu().numpy()
        masks = masks.cpu().numpy()

        planar_depth_map = copy.deepcopy(depth.cpu().numpy())

        for mask in masks:
            m_points = points[:, mask].T
            lstsq_res = np.linalg.lstsq(m_points, np.ones((m_points.shape[0], 1)))
            lst_plane_param = lstsq_res[0].squeeze(axis=-1)
            n1, n2, n3 = lst_plane_param

            m_planar_depth = (1 - n1 * m_points[:, 0] - n2 * m_points[:, 1]) / n3
            planar_depth_map[mask] = m_planar_depth

        planar_depth_map = torch.from_numpy(planar_depth_map).type_as(depth)

        return planar_depth_map

    def _forward_test(self, images, targets):
        images['ref_img'] = to_image_list(images['ref_img'])
        ref_img_tensors = images['ref_img'].tensors.cuda()

        images['src_img'] = to_image_list(images['src_img'])
        src_img_tensors = images['src_img'].tensors.cuda()

        ref_features = self.backbone(ref_img_tensors)
        src_features = self.backbone(src_img_tensors)

        ref_bot_feat = ref_features[0]
        src_bot_feat = src_features[0]

        hypos = images['depth_hypos'].unsqueeze(dim=0).cuda()
        pred_depth_map, refined_depth_map = self.stereo_net(ref_bot_feat, src_bot_feat, hypos, ref_img_tensors, [targets], is_test=True)

        if not self.wo_det_loss:
            proposals, _ = self.rpn(images['ref_img'], ref_features)

            attn_feats = None
            _, result, _ = self.roi_heads(ref_features, attn_feats, proposals, targets=targets)

        else:
            result = BoxList(torch.zeros(1, 4), (640, 480), mode='xyxy')
            result = [result]

        assert len(result) == 1, 'Currently only support test one image at a time'

        result[0].add_field('depth', pred_depth_map.squeeze(0).squeeze(0))

        if refined_depth_map is not None:
            result[0].add_field('refined_depth', refined_depth_map.squeeze(0).squeeze(0))

        if not self.wo_det_loss:
            planar_depth_map = self.fit_planes(result, targets)
            result[0].add_field('planar_depth', planar_depth_map)

        return result
