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
# you can ommit the "third_party.maskrcnn_main." for simplicity;
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn


""" load our own modules """
from src.models.roi_heads.roi_heads import build_roi_heads
from ..depth import build_depth
from ..stereo import build_stereo


class PlaneStereo_Finetune(nn.Module):
    def __init__(self, cfg):
        super(PlaneStereo_Finetune, self).__init__()

        self.cfg = cfg

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.wo_depth_head = cfg.MODEL.STEREO.WO_SINGLE_DEPTH

        if not self.wo_depth_head:
            self.depth_head = build_depth(cfg)

        self.stereo_net = build_stereo(cfg)

        self.with_loss_term_uncertainty = cfg.MODEL.STEREO.WITH_LOSS_TERM_UNCERTAINTY
        self.loss_term_num = cfg.MODEL.STEREO.LOSS_TERM_NUM

        if self.with_detection_consistency:
            self.loss_term_num += 5

        if self.with_loss_term_uncertainty:
            self.loss_term_uncert = nn.Parameter(torch.rand(self.loss_term_num).to('cuda'), requires_grad=True)
            torch.nn.init.constant_(self.loss_term_uncert, -1.0)

        # also run the detection head for src img
        self.infer_src_img = cfg.MODEL.STEREO.INFER_SRC_IMG

        self.with_instance_planar_depth_loss = cfg.MODEL.STEREO.INSTANCE_PLANAR_DEPTH_LOSS
        self.with_pred_instance_planar_depth_loss = cfg.MODEL.STEREO.PRED_INSTANCE_PLANAR_DEPTH_LOSS

        if self.with_pred_instance_planar_depth_loss:
            self.use_pixel_gt_plane_map = cfg.MODEL.STEREO.WITH_PIXEL_GT_PLANE_MAP

        # using gt or pred mask cannot be activated at the same time
        assert not (self.with_instance_planar_depth_loss and self.with_pred_instance_planar_depth_loss)

        self.finetune_depth_on = cfg.MODEL.STEREO.FINETUNE_DEPTH_ON
        self.ft_det_loss_weight = cfg.MODEL.STEREO.FINETUNE_DET_LOSS_WEIGHT

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

    def forward_test_during_training(self, images, max_det=10, use_src=False, no_grad=True):
        def forward_images(model, images, use_src):
            results = []
            model.eval()
            for i in range(len(images)):
                if use_src:
                    img = to_image_list(images[i]['src_img'])

                else:
                    img = to_image_list(images[i]['ref_img'])

                features = model.backbone(img.tensors)

                proposals, _ = model.rpn(img, features)

                if not model.wo_depth_head:
                    _, attn_feats, _ = model.depth_head(features)

                else:
                    attn_feats = None

                _, result, _ = model.roi_heads(features, attn_feats, proposals, targets=None)
                assert len(result) == 1

                results.append(result[0][:max_det])

            model.train()

            return results

        if no_grad:
            with torch.no_grad():
                results = forward_images(self, images, use_src=use_src)

        else:
            results = forward_images(self, images, use_src=use_src)

        return results

    def warp_src_to_tgt(self, src_results, targets):
        warped_masks = []
        warped_labels = []

        for i in range(len(src_results)):
            src_result = src_results[i]
            src_masks = src_result.get_field('mask').cuda()

            src_grid = targets[i].get_field('img_src_grids').unsqueeze(dim=0).repeat(src_masks.size(0), 1, 1, 1)

            warped_src_masks = F.grid_sample(src_masks.float(), src_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            warped_src_masks = warped_src_masks > 0.5

            warped_src_labels = src_result.get_field('labels')

            warped_masks.append(warped_src_masks)
            warped_labels.append(warped_src_labels)

        return warped_masks, warped_labels

    def get_instance_plane_depth_loss(self, pred_plane_map, targets):
        gt_masks = [target.get_field('masks').instances.masks.to('cuda') for target in targets]
        camera_grid = [target.get_field('img_camera_grid') for target in targets]

        h, w = pred_plane_map.size()[-2:]

        pred_depth_maps = []
        target_depth_maps = torch.stack([target.get_field('depth') for target in targets], dim=0)

        for idx, (img_plane_map, img_gt_masks, img_camera_grid) in enumerate(zip(pred_plane_map, gt_masks, camera_grid)):
            ins_planes = []
            for gt_mask in img_gt_masks:
                ins_planes.append(torch.mean(-img_plane_map[..., gt_mask], dim=-1))

            ins_planes = torch.stack(ins_planes)

            plane_piece_depths = (1. / (ins_planes @ img_camera_grid.view(3, -1))).view(img_gt_masks.size(0), h, w)
            pixel_plane_depth_map = (1. / torch.sum(-img_plane_map * img_camera_grid, dim=0)).view(h, w)

            img_plane_mask = img_gt_masks.sum(0) > 0
            plane_piece_depths = plane_piece_depths * img_gt_masks

            plane_piece_depths[plane_piece_depths < 1e-4] = 1e4
            plane_depth_map = torch.min(plane_piece_depths, dim=0)[0] * img_plane_mask + pixel_plane_depth_map * (~img_plane_mask)

            max_depth = torch.max(target_depth_maps[idx])

            plane_depth_map = plane_depth_map.clamp(min=0.5, max=max_depth)
            pred_depth_maps.append(plane_depth_map)

        pred_depth_maps = torch.stack(pred_depth_maps, dim=0)
        valid_mask = target_depth_maps > 1e-4

        loss = torch.mean(torch.abs(pred_depth_maps[valid_mask] - target_depth_maps[valid_mask]))
        loss = loss * self.cfg.MODEL.STEREO.PLANAR_DEPTH_LOSS_WEIGHT

        loss_dict = {'loss_instance_planar_depth': loss}

        return loss_dict

    def get_pred_instance_plane_depth_loss(self, pred_plane_map, results, targets, loss_mask=None):
        soft_masks = [result.get_field('soft_mask').squeeze(dim=1).to('cuda') for result in results]

        camera_grid = [target.get_field('img_camera_grid') for target in targets]
        h, w = pred_plane_map.size()[-2:]

        pred_depth_maps = []
        target_depth_maps = torch.stack([target.get_field('depth') for target in targets], dim=0)

        mask_nums = []

        for idx, (img_plane_map, img_soft_masks, img_camera_grid) in enumerate(zip(pred_plane_map, soft_masks, camera_grid)):
            ins_planes = []

            pixel_plane_depth_map = (1. / (torch.sum(-img_plane_map * img_camera_grid, dim=0))).view(h, w)

            if len(img_soft_masks) > 0:
                for soft_mask in img_soft_masks:
                    if loss_mask is not None:
                        img_loss_mask = loss_mask[idx]
                        soft_pooled_plane = (-img_plane_map[:, img_loss_mask] * soft_mask[img_loss_mask]).sum(-1) / (torch.sum(soft_mask[img_loss_mask]) + 1e-4)
                    else:
                        soft_pooled_plane = (-img_plane_map * soft_mask).sum(-1).sum(-1) / (torch.sum(soft_mask) + 1e-4)

                    ins_planes.append(soft_pooled_plane)

                ins_planes = torch.stack(ins_planes)
                plane_piece_depths = (1. / (ins_planes @ img_camera_grid.view(3, -1))).view(ins_planes.size(0), h, w)

                img_plane_mask = (img_soft_masks > 0.5).sum(0) > 0
                img_bin_masks = img_soft_masks > 0.5

                plane_piece_depths = plane_piece_depths * img_bin_masks
                plane_piece_depths[plane_piece_depths < 1e-4] = 1e4

                plane_depth_map = torch.min(plane_piece_depths, dim=0)[0] * img_plane_mask + pixel_plane_depth_map * (~img_plane_mask)

            else:
                plane_depth_map = pixel_plane_depth_map

            max_depth = torch.max(target_depth_maps[idx])

            plane_depth_map = plane_depth_map.clamp(min=0.5, max=max_depth)
            pred_depth_maps.append(plane_depth_map)

            mask_nums.append(img_soft_masks.size(0))

        pred_depth_maps = torch.stack(pred_depth_maps, dim=0)
        valid_mask = target_depth_maps > 1e-4

        if loss_mask is not None:
            valid_mask = valid_mask * loss_mask

        loss = torch.mean(torch.abs(pred_depth_maps[valid_mask] - target_depth_maps[valid_mask]))
        loss = loss * self.cfg.MODEL.STEREO.PLANAR_DEPTH_LOSS_WEIGHT

        # sometimes the pred_depth_maps has nan, so we deal with it separately
        if (pred_depth_maps != pred_depth_maps).any() or (loss != loss).any():
            return None

        loss_dict = {'loss_pred_instance_planar_depth': loss}

        return loss_dict

    def _forward_train(self, images, targets, infos=None):
        ref_img_tensors = torch.cat([image['ref_img'].tensors for image in images], dim=0)
        src_img_tensors = torch.cat([image['src_img'].tensors for image in images], dim=0)

        ref_features = self.backbone(ref_img_tensors)
        src_features = self.backbone(src_img_tensors)

        # get the coarse layer feature from fpn levels,
        # which should be 1/4 H * 1/4 W resolution
        ref_bot_feat = ref_features[0]
        src_bot_feat = src_features[0]

        homo_grids = torch.stack([image['homo_grid'] for image in images], dim=0)
        hypos = torch.stack([image['hypos'] for image in images], dim=0)

        pred_plane_map, refined_plane_map, pred_planar_depth, refined_pred_planar_depth, uncertainty_map, plane_losses = self.stereo_net(ref_bot_feat, src_bot_feat, homo_grids, hypos, ref_img_tensors, targets)

        if self.with_instance_planar_depth_loss:
            instance_plane_depth_loss = self.get_instance_plane_depth_loss(pred_plane_map, targets)
            plane_losses.update(instance_plane_depth_loss)

        if self.with_pred_instance_planar_depth_loss:
            results = self.forward_test_during_training(images, use_src=False, no_grad=False)
            if self.use_pixel_gt_plane_map:
                gt_pixel_plane_map = torch.stack([t.get_field('pixel_n_div_d_map') for t in targets])
                gt_pixel_plane_mask = torch.stack([t.get_field('pixel_plane_mask') for t in targets])

                # [3, h, w]
                gt_pixel_plane_map = gt_pixel_plane_map.permute(0, 3, 1, 2)
                gt_pixel_plane_mask = gt_pixel_plane_mask.bool()

                pred_instance_plane_depth_loss = self.get_pred_instance_plane_depth_loss(gt_pixel_plane_map, results, targets, loss_mask=gt_pixel_plane_mask)

            else:
                pred_instance_plane_depth_loss = self.get_pred_instance_plane_depth_loss(pred_plane_map, results, targets)

            if pred_instance_plane_depth_loss is None:
                pred_instance_plane_depth_loss = plane_losses['loss_pixel_planar_depth']
                pred_instance_plane_depth_loss = {'loss_pred_instance_planar_depth': pred_instance_plane_depth_loss}

            plane_losses.update(pred_instance_plane_depth_loss)

        losses = OrderedDict()
        losses.update(plane_losses)

        image_lists = to_image_list(ref_img_tensors)
        proposals, proposal_losses = self.rpn(image_lists, ref_features, targets)

        attn_feats = None
        _, _, detector_losses = self.roi_heads(ref_features, attn_feats, proposals, targets)

        losses.update(proposal_losses)
        losses.update(detector_losses)

        if self.finetune_depth_on:
            detection_terms = ['loss_box_reg', 'loss_classifier', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg']

            for key, val in losses.items():
                if key in detection_terms:
                    losses[key] = losses[key] * self.ft_det_loss_weight

        return losses
