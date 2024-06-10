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
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn


""" load our own modules """
from src.models.roi_heads.roi_heads import build_roi_heads
from src.utils.colormap import ColorPalette
from ..depth import build_depth
from ..refine import build_refine


class PlaneRCNNRefine(nn.Module):
    def __init__(self, cfg):
        super(PlaneRCNNRefine, self).__init__()

        self.cfg = cfg

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.depth_head = build_depth(cfg)

        self.refine_size = cfg.MODEL.REFINE.REFINE_SIZE
        self.default_size = cfg.MODEL.REFINE.DEFAULT_SIZE

        self.color_mapping = ColorPalette(numColors=100).getColorMap(returnTuples=True)

        self.with_mask_refine_loss = cfg.MODEL.REFINE.WITH_MASK_REFINE_LOSS

        if self.with_mask_refine_loss:
            self.refine_model = build_refine(cfg)

        self.with_warping_loss = cfg.MODEL.REFINE.WITH_WARPING_LOSS

        if self.with_warping_loss:
            assert not cfg.MODEL.ROI_BOX_HEAD.WO_NORMAL_HEAD

        self.use_soft_mask = cfg.MODEL.REFINE.USE_SOFT_MASK

    def make_camera_grid(self, intrinsic, depth):
        h, w = depth.size()
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

        grid = torch.ones(3, h, w)
        grid[0, ...] = xs
        grid[1, ...] = ys

        grid = intrinsic[:3, :3].inverse() @ grid.contiguous().view(3, -1).type_as(intrinsic)
        grid = grid.view(3, h, w).to(depth.device)

        points = grid * depth

        return grid, points

    def plane_to_depth(self, result, target, for_refine=False, min_depth=None, max_depth=None):
        if max_depth is None:
            max_depth = torch.max(target.get_field('depth'))

        if min_depth is None:
            valid_mask = target.get_field('depth') > 1e-4
            min_depth = torch.min(target.get_field('depth')[valid_mask])

        np_depth = result.get_field('depth').squeeze().clamp(min=min_depth, max=max_depth)
        masks = result.get_field('mask').squeeze(dim=1).to(np_depth.device)

        h, w = np_depth.size()[-2:]
        intrinsic = target.get_field('intrinsic')[:3, :3]

        camera_grids, points = self.make_camera_grid(intrinsic, np_depth)

        if (not for_refine and masks.size(0) == 0) or (for_refine and masks.size(0) == 1):
            # use non-planar coords to take place
            depth = np_depth
            plane_xyz = points.unsqueeze(dim=0)

        else:
            pred_normals = result.get_field('normal_res')
            pred_normals = pred_normals / (torch.norm(pred_normals, dim=-1, keepdim=True) + 1e-10)

            offsets = ((pred_normals @ points.view(3, -1)).view(masks.size(0), h, w) * masks).sum(-1).sum(-1) / masks.sum(-1).sum(-1)

            plane_piece_depths = offsets.unsqueeze(dim=-1).unsqueeze(dim=-1) / ((pred_normals @ camera_grids.view(3, -1)).view(masks.size(0), h, w) + 1e-10)

            plane_piece_depths = plane_piece_depths * masks

            plane_piece_depths[plane_piece_depths < 1e-4] = 1e4
            plane_depth_map = torch.min(plane_piece_depths, dim=0)[0]

            pred_plane_mask = masks.sum(0) > 0
            plane_depth_map = plane_depth_map * pred_plane_mask

            depth = plane_depth_map * pred_plane_mask + np_depth * (~pred_plane_mask)
            depth = depth.clamp(min=min_depth, max=max_depth)

            plane_xyz = plane_piece_depths.unsqueeze(dim=1).clamp(min=min_depth, max=max_depth) * camera_grids.unsqueeze(dim=0)

        return depth, plane_xyz

    def forward(self, images, targets=None, infos=None):
        def set_bn_eval(m):
            classname = m.__class__.__name__

            if classname.find('BatchNorm') != -1:
                m.eval()

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            self.apply(set_bn_eval)
            return self._forward_train(images, targets)

        else:
            return self._forward_test(images, targets)

    def get_mask_iou(self, mask_set1, mask_set2):
        mask_set1 = mask_set1.unsqueeze(dim=1)
        mask_set2 = mask_set2.unsqueeze(dim=0)

        i = (mask_set1 * mask_set2).sum(-1).sum(-1)
        u = ((mask_set1 + mask_set2) > 0).sum(-1).sum(-1)

        iou = i / (u + 1e-4)

        return iou

    def match_mask_targets_to_proposals(self, pred_masks, target_masks, iou_thresh=0.5):
        pred_masks_bool = pred_masks > 0.5
        target_masks_bool = target_masks > 0.5
        # [N, M]
        mask_iou = self.get_mask_iou(pred_masks_bool, target_masks_bool)

        # the gt idx matched for each pred
        matched_idxs = mask_iou.argmax(dim=-1)
        pos_idxs = mask_iou.max(dim=-1)[0] > iou_thresh

        return matched_idxs, pos_idxs

    def visualize_pred_target(self, ori_img, img_path, initial_pred_masks, pred_masks, target_masks):
        assert len(pred_masks) == len(target_masks)

        pred_masks_bool = torch.sigmoid(pred_masks) > 0.5
        target_masks_bool = target_masks > 0.5

        initial_pred_masks_bool = initial_pred_masks.bool().cpu().numpy()
        pred_masks_bool = pred_masks_bool.cpu().numpy()[1:]
        target_masks_bool = target_masks_bool.cpu().numpy()[1:]

        initial_pred_vis = np.zeros(ori_img.shape)
        pred_vis = np.zeros(ori_img.shape)
        target_vis = np.zeros(ori_img.shape)

        for idx, (initial_pred_mask, pred_mask, target_mask) in enumerate(zip(initial_pred_masks_bool, pred_masks_bool, target_masks_bool)):
            if idx == 0:
                continue

            initial_pred_vis[initial_pred_mask] = self.color_mapping[idx]
            pred_vis[pred_mask] = self.color_mapping[idx]
            target_vis[target_mask] = self.color_mapping[idx]

        initial_pred_vis = initial_pred_vis * 0.5 + ori_img * 0.5
        pred_vis = pred_vis * 0.5 + ori_img * 0.5
        target_vis = target_vis * 0.5 + ori_img * 0.5

        vis = np.hstack([initial_pred_vis, pred_vis, target_vis])

        save_dir = 'debug_refinement'
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        cv2.imwrite(osp.join(save_dir, img_path.split('/')[-1]), vis)

    def get_refine_loss(self, ori_img, img_path, initial_pred, refined_pred, target, debug=False):
        target_mask = target.get_field('masks')

        matched_target_idxs, pos_pred_idxs = self.match_mask_targets_to_proposals(initial_pred, target_mask)
        matched_target_masks = target_mask[matched_target_idxs[pos_pred_idxs]]

        refined_pred = torch.cat([refined_pred[0:1], refined_pred[1:][pos_pred_idxs]], dim=0)

        # add non-planar pred supervision gt
        matched_target_masks = torch.cat([1 - target_mask.sum(0, keepdim=True), matched_target_masks], dim=0)

        refine_mask_loss = F.binary_cross_entropy_with_logits(
            refined_pred, matched_target_masks)

        if debug:
            initial_pred = initial_pred[pos_pred_idxs]
            self.visualize_pred_target(ori_img, img_path, initial_pred, refined_pred, matched_target_masks)

        return refine_mask_loss

    def prepare_refine(self, result, target):
        mask = result.get_field('mask')

        if mask.size(0) > 0:
            mask = F.interpolate(mask.float(), size=self.refine_size, mode='bilinear', align_corners=True)

            if self.use_soft_mask:
                soft_mask = result.get_field('soft_mask')
                soft_mask = F.interpolate(soft_mask, size=self.refine_size, mode='bilinear', align_corners=True)

        else:
            mask = torch.zeros(1, 1, *self.refine_size)
            if self.use_soft_mask:
                soft_mask = mask.clone()

        result.add_field('mask', mask)

        if self.use_soft_mask:
            result.add_field('soft_mask', soft_mask)

        depth = result.get_field('depth')
        depth = F.interpolate(depth.unsqueeze(dim=1), size=self.refine_size, mode='bilinear', align_corners=True)
        result.add_field('depth', depth.squeeze(dim=1))

        intrinsic = target.get_field('intrinsic')
        intrinsic[0][0] = intrinsic[0][0] * self.refine_size[1] / self.default_size[1]
        intrinsic[0][2] = intrinsic[0][2] * self.refine_size[1] / self.default_size[1]

        intrinsic[1][1] = intrinsic[1][1] * self.refine_size[0] / self.default_size[0]
        intrinsic[1][2] = intrinsic[1][2] * self.refine_size[0] / self.default_size[0]

        target.add_field('intrinsic', intrinsic)

        mask_target = target.get_field('masks').instances.masks
        mask_target = F.interpolate(mask_target.unsqueeze(dim=1).float(), size=self.refine_size, mode='nearest')

        # overwrite
        target.add_field('masks', mask_target.squeeze(dim=1).cuda())

        return

    def resize_back(self, mask):
        mask = F.interpolate(mask.unsqueeze(dim=1).float(), self.default_size, mode='bilinear', align_corners=True)
        # turn to bool
        mask = torch.sigmoid(mask)
        bin_mask = mask > 0.5

        return bin_mask

    def warp_src_to_ref(self, src_planar_depth, image, target, use_gt_depth=True, debug=False):
        intrinsic = target.get_field('intrinsic')

        if use_gt_depth:
            depth = target.get_field('depth')
            depth_valid_mask = depth > 1e-4
            _, ref_points = self.make_camera_grid(intrinsic, depth)

            _, pred_src_points = self.make_camera_grid(intrinsic, src_planar_depth)

        else:
            raise NotImplementedError

        ref_pose = target.get_field('ref_pose')
        src_pose = target.get_field('src_pose')

        _, h, w = ref_points.size()

        homo_ref_points = torch.ones(4, h, w).to(ref_points.device)
        homo_ref_points[:3, ...] = ref_points

        src_points = src_pose.inverse() @ ref_pose @ homo_ref_points.view(4, -1)
        src_points = (intrinsic[:3, :3] @ src_points[:3, ...]).view(3, h, w)

        src_xys = src_points[:2, ...] / src_points[-1:, ...]

        # [h, w, 2]
        src_xys = src_xys.permute(1, 2, 0)

        workspace = torch.tensor([(w-1) / 2, (h-1) / 2]).to(src_xys.device).unsqueeze(dim=0).unsqueeze(dim=0)
        src_xys = src_xys / workspace - 1
        valid_mask = depth_valid_mask * (src_xys[..., 0] >= -1) * (src_xys[..., 1] >= -1) * (src_xys[..., 0] < 1) * (src_xys[..., 1] < 1)

        if debug:
            ori_src_img = np.asarray(image['ori_src_img'])
            src_img = torch.from_numpy(ori_src_img).float().permute(2, 0, 1).cuda()
            ref_img = np.asarray(image['ori_ref_img'])
            ref_img_id = image['ref_path'].split('/')[-1]

            warped_src_img = F.grid_sample(src_img.unsqueeze(dim=0), src_xys.unsqueeze(dim=0).type_as(src_img), mode='bilinear', padding_mode='zeros', align_corners=True)

            warped_src_img = warped_src_img.squeeze(dim=0).permute(1, 2, 0).cpu().numpy()

            save_dir = 'debug_warping_loss'
            if not osp.exists(save_dir):
                os.makedirs(save_dir)

            vis = cv2.imwrite(osp.join(save_dir, ref_img_id), np.hstack([ref_img, ori_src_img, warped_src_img]))

        warped_src_points = F.grid_sample(pred_src_points.unsqueeze(dim=0), src_xys.unsqueeze(dim=0), mode='bilinear', padding_mode='zeros', align_corners=True)
        warped_src_points = warped_src_points.squeeze(dim=0)

        return warped_src_points, valid_mask

    def transform_to_ref(self, warped_src_xyz, target):
        ref_pose = target.get_field('ref_pose')
        src_pose = target.get_field('src_pose')

        _, h, w = warped_src_xyz.size()

        warped_src_xyz = warped_src_xyz.view(3, -1)
        homo_warped_src_xyz = torch.ones(4, warped_src_xyz.size(-1)).to(warped_src_xyz.device)
        homo_warped_src_xyz[:3, :] = warped_src_xyz

        homo_warped_src_xyz_in_ref = ref_pose.inverse() @ src_pose @ homo_warped_src_xyz
        warped_src_xyz_in_ref = homo_warped_src_xyz_in_ref[:3, :].view(3, h, w)

        return warped_src_xyz_in_ref

    # masked l1 loss
    def planercnn_warping_loss(self, ref_depth, transformed_warped_src_depth, valid_mask):
        return (torch.abs(ref_depth[valid_mask] - transformed_warped_src_depth[valid_mask])).mean()

    def _forward_train(self, images, targets, max_refine_size=7):
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

        if self.with_mask_refine_loss:
            initial_results = []

        if self.with_warping_loss:
            warping_losses = []

        self.eval()
        for image, target in zip(images, targets):
            initial_result = self._forward_test_during_training(image, target, use_src=False)
            initial_result = initial_result[0]

            if self.with_warping_loss:
                planar_depth, plane_xyz = self.plane_to_depth(initial_result, target, for_refine=False)

                src_initial_result = self._forward_test_during_training(image, target, use_src=True)
                src_initial_result = src_initial_result[0]

                src_planar_depth, src_plane_xyz = self.plane_to_depth(src_initial_result, target, for_refine=False)
                warped_src_planar_depth, valid_mask = self.warp_src_to_ref(src_planar_depth, image, target, use_gt_depth=True, debug=False)

                warped_src_xyz_in_ref = self.transform_to_ref(warped_src_planar_depth, target)
                warped_src_depth_in_ref = warped_src_xyz_in_ref[-1, ...]

                if valid_mask.sum() > 0.25 * 640 * 480:
                    warping_loss = self.planercnn_warping_loss(planar_depth, warped_src_depth_in_ref, valid_mask)
                    warping_losses.append(warping_loss)

            if self.with_mask_refine_loss:
                if max_refine_size > 0:
                    initial_result = initial_result[:max_refine_size]

                self.prepare_refine(initial_result, target)
                planar_depth, plane_xyz = self.plane_to_depth(initial_result, target, for_refine=True, min_depth=0.1, max_depth=10)

                initial_result.add_field('planar_depth', planar_depth)
                initial_result.add_field('plane_xyz', plane_xyz)
                initial_results.append(initial_result)

        self.train()

        if self.with_mask_refine_loss:
            img_tensors = F.interpolate(img_tensors, size=self.refine_size, mode='bilinear', align_corners=True)
            ori_imgs = np.stack([cv2.resize(np.asarray(images[i]['ori_ref_img'])[..., ::-1], (self.refine_size[1], self.refine_size[0]), interpolation=cv2.INTER_LINEAR) for i in range(len(images))], axis=0)
            img_paths = [image['ref_path'] for image in images]

            refine_losses = []
            initial_preds, refined_preds = [], []

            for ori_img, img_path, img_t, result, target in zip(ori_imgs, img_paths, img_tensors, initial_results, targets):
                refined_pred = self.refine_model(img_t, result)
                # used for match gts
                initial_pred = result.get_field('mask').bool().squeeze(dim=1).cuda()

                refine_loss = self.get_refine_loss(ori_img, img_path, initial_pred, refined_pred, target)
                refine_losses.append(refine_loss)

                initial_preds.append(initial_pred)
                refined_preds.append(refined_pred)

            refine_loss = torch.stack(refine_losses).mean()
            refine_loss = {
                'loss_refine_mask': refine_loss
            }

            losses.update(refine_loss)

        if self.with_warping_loss and len(warping_losses) > 0:
            warping_loss = torch.stack(warping_losses).mean()
            warping_loss = {
                'loss_warping': warping_loss
            }
            losses.update(warping_loss)

        return losses

    def _forward_test_during_training(self, images, targets, use_src=False):
        if use_src:
            images['src_img'] = to_image_list(images['src_img'])
            features = self.backbone(images['src_img'].tensors.cuda())

            proposals, _ = self.rpn(images['src_img'], features)

        else:
            images['ref_img'] = to_image_list(images['ref_img'])
            features = self.backbone(images['ref_img'].tensors.cuda())

            proposals, _ = self.rpn(images['ref_img'], features)

        depth, attn_feats, _ = self.depth_head(features)

        assert 'anchor_normals' in targets.fields()
        _, result, _ = self.roi_heads(features, attn_feats, proposals, targets=targets)

        assert len(result) == 1, 'Currently only support test one image at a time'
        result[0].add_field('depth', depth)

        return result

    def _forward_test(self, images, targets):
        images['ref_img'] = to_image_list(images['ref_img'])
        img_tensors = images['ref_img'].tensors.cuda()
        features = self.backbone(img_tensors)

        proposals, _ = self.rpn(images['ref_img'], features)

        depth, attn_feats, _ = self.depth_head(features)

        assert 'anchor_normals' in targets.fields()
        _, initial_result, _ = self.roi_heads(features, attn_feats, proposals, targets=targets)
        assert len(initial_result) == 1, 'Currently only support test one image at a time'

        initial_result[0].add_field('depth', depth)

        if not self.with_mask_refine_loss:
            return initial_result

        initial_result_copy = copy.deepcopy(initial_result)
        targets_copy = copy.deepcopy(targets)
        self.prepare_refine(initial_result_copy[0], targets_copy)

        planar_depth, plane_xyz = self.plane_to_depth(initial_result_copy[0], targets_copy,
                                                      for_refine=True, min_depth=0.1, max_depth=10)

        initial_result_copy[0].add_field('planar_depth', planar_depth)
        initial_result_copy[0].add_field('plane_xyz', plane_xyz)

        img_tensors = F.interpolate(img_tensors, size=self.refine_size, mode='bilinear', align_corners=True)
        refined_mask = self.refine_model(img_tensors, initial_result_copy[0])

        # except for non-planar
        refined_mask = refined_mask[1:]

        resized_refined_mask = self.resize_back(refined_mask)
        initial_result[0].add_field('refined_mask', resized_refined_mask)

        return initial_result
