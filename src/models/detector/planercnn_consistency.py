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


class PlaneRCNNConsistency(nn.Module):
    def __init__(self, cfg):
        super(PlaneRCNNConsistency, self).__init__()

        self.cfg = cfg

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
            self.apply(set_bn_eval)
            return self._forward_train(images, targets, infos)

        else:
            return self._forward_test(images, targets)

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

    def masks_to_bboxes(self, masks, labels, min_area=500):
        valid_idxs = []
        masks = masks.cpu().numpy()
        labels = labels.cpu().numpy()

        bboxes = []

        for idx, mask in enumerate(masks):
            if mask.sum() < min_area:
                continue

            fg = np.where(mask > 0)
            bbox = [
                np.min(fg[1]),
                np.min(fg[0]),
                np.max(fg[1]) + 1,
                np.max(fg[0]) + 1
            ]

            bbox = list(map(int, bbox))
            bboxes.append(bbox)
            valid_idxs.append(idx)

        if len(valid_idxs) == 0:
            return None, None, None

        bboxes = np.asarray(bboxes)

        valid_idxs = np.array(valid_idxs)
        labels = labels[valid_idxs]
        masks = masks[valid_idxs]

        return masks, bboxes, labels

    def make_target(self, warped_src_masks, warped_src_labels, targets):
        pseudo_targets = []
        for idx, (img_masks, img_labels) in enumerate(zip(warped_src_masks, warped_src_labels)):
            if len(img_masks) == 0:
                target = targets[idx]
                target = target.copy_with_fields(['masks', 'labels'])

            else:
                masks, bboxes, labels = self.masks_to_bboxes(img_masks, img_labels)

                if masks is None or len(masks) == 0:
                    target = targets[idx]
                    target = target.copy_with_fields(['masks', 'labels'])

                else:
                    target = BoxList(torch.from_numpy(bboxes), (640, 480), mode='xyxy')

                    masks = SegmentationMask(torch.from_numpy(masks.squeeze(axis=1)), (640, 480), mode='mask')
                    target.add_field('masks', masks)

                    target.add_field('labels', torch.from_numpy(labels).long())

            target = target.to('cuda')
            pseudo_targets.append(target)

        return pseudo_targets

    def _forward_train(self, images, targets, infos=None):
        img_tensors = torch.cat([image['ref_img'].tensors for image in images], dim=0)

        features = self.backbone(img_tensors)
        image_lists = to_image_list(torch.cat([image['ref_img'].tensors for image in images], dim=0))

        _, attn_feats, depth_loss = self.depth_head(features, targets)
        # here actually only the image size is needed
        proposals, proposal_losses = self.rpn(image_lists, features, targets)

        _, _, detector_losses = self.roi_heads(features, attn_feats, proposals, targets)

        losses = {}
        losses.update(proposal_losses)
        losses.update(depth_loss)
        losses.update(detector_losses)

        src_results = self.forward_src_during_training(images)
        warped_masks, warped_labels = self.warp_src_to_tgt(src_results, targets)

        pseudo_targets = self.make_target(warped_masks, warped_labels, targets)
        masked_img_tensors = torch.cat([image['masked_img'].tensors for image in images], dim=0)

        features = self.backbone(masked_img_tensors)
        image_lists = to_image_list(masked_img_tensors)

        proposals, proposal_losses = self.rpn(image_lists, features, pseudo_targets)

        # here depth feats have no effect actually
        _, _, detector_losses = self.roi_heads(features, attn_feats, proposals, pseudo_targets)

        consistency_losses = {}

        for key, loss in proposal_losses.items():
            new_key = key.replace('loss', 'loss_pseudo')
            consistency_losses[new_key] = loss

        for key, loss in detector_losses.items():
            new_key = key.replace('loss', 'loss_pseudo')
            consistency_losses[new_key] = loss

        losses.update(consistency_losses)

        return losses

    def forward_src_during_training(self, images, max_det=15):
        src_results = []

        self.eval()
        with torch.no_grad():
            for i in range(len(images)):
                src_img = to_image_list(images[i]['src_img'])
                src_features = self.backbone(src_img.tensors)

                proposals, _ = self.rpn(src_img, src_features)
                _, attn_feats, _ = self.depth_head(src_features)

                _, src_result, _ = self.roi_heads(src_features, attn_feats, proposals, targets=None)
                assert len(src_result) == 1

                src_results.append(src_result[0][:max_det])

        self.train()

        return src_results

    def _forward_test(self, images, targets):
        images['ref_img'] = to_image_list(images['ref_img'])
        img_tensors = images['ref_img'].tensors.cuda()
        features = self.backbone(img_tensors)

        proposals, _ = self.rpn(images['ref_img'], features)

        depth, attn_feats, _ = self.depth_head(features)

        assert 'anchor_normals' in targets.fields()
        _, result, _ = self.roi_heads(features, attn_feats, proposals, targets=targets)

        assert len(result) == 1, 'Currently only support test one image at a time'
        result[0].add_field('depth', depth)

        return result
