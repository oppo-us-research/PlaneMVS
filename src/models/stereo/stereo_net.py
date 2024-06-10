"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StereoNet(nn.Module):
    def __init__(self, cfg, in_channels=256):
        super(StereoNet, self).__init__()
        self.cfg = cfg

        self.n_hypos = int(cfg.MODEL.STEREO.N_HYPOS_PER_AXIS) ** 3

        # concatenate the features by default
        self.feature_type = cfg.MODEL.STEREO.FEATURE_TYPE

        # reduce the feature channel by 1x1 conv to save memory
        self.feat_reduction = nn.Conv2d(in_channels, self.cfg.MODEL.STEREO.FEATURE_REDUCT_CHANNELS, 1, 1, 0)

        # our default setting
        if self.feature_type == 'concat':
            self.cost_reg_in_channels = 2 * self.cfg.MODEL.STEREO.FEATURE_REDUCT_CHANNELS

        # the setting in mvsnet
        elif self.feature_type == 'var':
            self.cost_reg_in_channels = self.cfg.MODEL.STEREO.FEATURE_REDUCT_CHANNELS

        else:
            raise NotImplementedError

        # use group-norm to normalize the stereo net
        self.apply_gn = cfg.MODEL.STEREO.APPLY_GN

        # whether to add a refinement network after initial plane prediction
        self.apply_refine = cfg.MODEL.STEREO.APPLY_REFINE

        # whether to supervise the depth for plane prediction
        self.with_pixel_depth_loss = cfg.MODEL.STEREO.WITH_PIXEL_DEPTH_LOSS

        self.pool = cfg.MODEL.STEREO.POOL_FEATURE

        self.make_cost_reg_mvsnet()

        # using raft upsampling method(convex combination) to upsample prediction from 1/8 to 1
        self.use_raft_upsample = cfg.MODEL.STEREO.RAFT_UPSAMPLE

        if self.apply_refine:
            self.make_refine_net(6, 32)

        # 1 / 4 img resolution by default(the bottom layer of fpn feature map)
        self.stereo_h = cfg.MODEL.STEREO.STEREO_H
        self.stereo_w = cfg.MODEL.STEREO.STEREO_W

        # img resolution
        self.img_h = cfg.MODEL.IMG_H
        self.img_w = cfg.MODEL.IMG_W

        if self.pool:
            self.stereo_h = self.stereo_h // 2
            self.stereo_w = self.stereo_w // 2

            self.pool_layer = nn.AvgPool2d((2, 2), stride=(2, 2))

        if self.use_raft_upsample: # True in our model;
            # RAFT (https://arxiv.org/abs/2003.12039) style convex upsampling;
            self.make_raft_upsample_mask()

        # stereo feature grids
        self.feat_grids = self.make_grids(self.stereo_w, self.stereo_h)
        # img grids
        self.img_grids = self.make_grids(self.img_w, self.img_h)
        # only fine-tune the depth when tuning on custom rgb-d video datasets
        self.finetune_wo_plane_loss = cfg.MODEL.STEREO.FINETUNE_WO_PLANE_LOSS

        self.with_plane_losses = cfg.MODEL.STEREO.WITH_PLANE_LOSS

        self._init_weights()

    # compute img grids
    def make_grids(self, w, h):
        xxs, yys = np.meshgrid(np.arange(w), np.arange(h))
        xys = np.ones((h, w, 3))
        xys[..., 0] = xxs
        xys[..., 1] = yys

        xys = torch.from_numpy(xys).to('cuda')
        xys.requires_grad = False

        return xys
    

    # compute camera grids
    def make_camera_grid(self, intrinsic, grid):
        h, w = grid.size()[:2]
        bs = intrinsic.size(0)
        camera_grid = intrinsic.inverse() @ grid.type_as(intrinsic).permute(2, 0, 1).view(3, -1)
        camera_grid = camera_grid.view(bs, 3, h, w)

        return camera_grid

    # initialize weights for stereo network
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def convgn_3d(self, in_planes, out_planes, kernel_size, stride, pad, num_groups=8):
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_planes))

    def conv_3d_relu(self, in_planes, out_planes, kernel_size, stride, pad):
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride),
            nn.ReLU(inplace=True))

    def conv_3d_leakyrelu(self, in_planes, out_planes, kernel_size, stride, pad):
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride)
        )

    def convgn_3d_relu(self, in_planes, out_planes, kernel_size, stride, pad, num_groups=8):
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_planes),
            nn.ReLU(inplace=True))

    def convgn_2d_relu(self, in_planes, out_planes, kernel_size=3, stride=1, pad=1, num_groups=8):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_planes),
            nn.ReLU(inplace=True))

    def conv_3d(self, in_planes, out_planes, kernel_size, stride, pad):
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride))

    def convbn_3d_o(self, in_planes, out_planes, kernel_size, stride, pad):
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
            nn.BatchNorm3d(out_planes))

    def convbn_3d_relu(self, in_planes, out_planes, kernel_size, stride, pad):
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True))

    # build stereo net as the network used in mvsnet paper
    def make_cost_reg_mvsnet(self):
        if self.apply_gn:
            conv_func = self.convgn_3d_relu

        else:
            conv_func = self.convbn_3d_relu

        self.conv0 = conv_func(self.cost_reg_in_channels, 8, 3, 1, 1)

        self.conv1 = conv_func(8, 16, 3, 2, 1)
        self.conv2 = conv_func(16, 16, 3, 1, 1)

        self.conv3 = conv_func(16, 32, 3, 2, 1)
        self.conv4 = conv_func(32, 32, 3, 1, 1)

        if self.pool:
            self.conv6 = conv_func(32, 32, 3, 1, 1)

        else:
            self.conv5 = conv_func(32, 64, 3, 2, 1)
            self.conv6 = conv_func(64, 64, 3, 1, 1)

        if self.apply_gn:
            if not self.pool:
                self.conv7 = nn.Sequential(
                    nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=32),
                    nn.ReLU(inplace=True)
                )

            self.conv9 = nn.Sequential(
                nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=16),
                nn.ReLU(inplace=True)
            )

            self.conv11 = nn.Sequential(
                nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=8),
                nn.ReLU(inplace=True)
            )

        else:
            if not self.pool:
                self.conv7 = nn.Sequential(
                    nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                    nn.BatchNorm3d(32),
                    nn.ReLU(inplace=True)
                )

            self.conv9 = nn.Sequential(
                nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True)
            )

            self.conv11 = nn.Sequential(
                nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True)
            )


        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    # build a light network to refine initial prediction
    def make_refine_net(self, in_planes, out_planes):
        if self.apply_gn:
            conv_func = self.convgn_2d_relu

        else:
            conv_func = self.convbn_2d_relu

        self.refine_conv1 = conv_func(in_planes, out_planes)
        self.refine_conv2 = conv_func(out_planes, out_planes)
        self.refine_conv3 = conv_func(out_planes, out_planes)

        self.refine_res = nn.Conv2d(out_planes, 3, 3, 1, 1)

    # feature warping
    def src_warp(self, src_coords, src_feat, padding_mode='zeros'):
        b, n, h, w, _ = src_coords.size()
        c = src_feat.size(1)
        src_coords = src_coords.view(b, n * h, w, 2)
        #print (f"??? {src_feat.dtype}, vs,  {src_coords.dtype}")
        warped_src_feat = F.grid_sample(src_feat, src_coords, mode='bilinear', padding_mode=padding_mode, align_corners=True)
        warped_src_feat = warped_src_feat.view(b, c, n, h, w)

        return warped_src_feat

    # homo_grids: (b, n, 3, 3), src_feat: (b, 32, 120, 160)
    def get_src_coords(self, batch_size, homo_grids, grids):
        # (b, 1, h, w, 3, 1)
        grids = grids.repeat(batch_size, 1, 1, 1).unsqueeze(dim=1).unsqueeze(dim=-1)
        # (b, n, 1, 1, 3, 3)
        homo_grids = homo_grids.unsqueeze(dim=2).unsqueeze(dim=3)
        # (b, n, 3, 3) -> (b, h, w, 3, 1) -> (b, n, h, w, 3, 1) -> (b, n, h, w, 3)
        src_coords = (homo_grids @ grids.type_as(homo_grids)).squeeze(dim=-1)
        # (b, n, h, w, 2)
        src_xys = src_coords[..., :2] / (src_coords[..., -1:] + 1e-10)

        h, w = src_xys.size(2), src_xys.size(3)

        workspace = torch.tensor([(w - 1) / 2, (h - 1) / 2]).to(src_xys.device)
        workspace = workspace.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # normalize the src grids
        src_xys = src_xys / workspace - 1

        return src_xys

    # soft-argmin to get plane map prediction
    def plane_regression(self, prob, plane_hypos):
        # (b, n, 120, 160) * (b,n,3)
        # = (b,n,120,160,1) * (b,n,1,1,3)
        # = (b, n, 120, 160, 3)
        # ===> reduce sum ==>  (b, 120, 160, 3)
        plane_map = torch.sum(prob.unsqueeze(dim=-1) * plane_hypos.unsqueeze(dim=-2).unsqueeze(dim=-2), dim=1)
        # (b, 120, 160, 3) -->permute--> (b, 3, 120, 160)
        plane_map = plane_map.permute(0, 3, 1, 2)
        return plane_map

    # masked l1 loss, only compute on valid pixels
    def plane_l1_loss(self, pred, target, mask, uncertainty=None):
        mask = (mask > 0.5).unsqueeze(dim=1).repeat(1, 3, 1, 1)

        assert uncertainty is None

        loss = torch.mean(torch.abs(pred[mask] - target[mask]))

        return loss

    # transform plane-map to depth-map then compute loss
    def pixel_depth_loss(self, pred_plane_map, targets, 
                         uncertainty=None, max_depth=10, 
                         for_refine=False, error_map=None
                         ):
        target_depth = torch.stack([t.get_field('depth') for t in targets], dim=0)
        mask = target_depth > 1e-4

        intrinsic = torch.stack([t.get_field('intrinsic') for t in targets], dim=0)[:, :3, :3]
        # (b, 3, h, w)
        camera_grid = self.make_camera_grid(intrinsic, self.img_grids)

        # fix the pixel-planar depth loss due to the sign of homography changed
        # (b,3,h,w)*(b,3,h,w) ==> reduced-sum ==> (b,h,w)
        plane_depth_map = 1 / torch.sum(-pred_plane_map * camera_grid, dim=1)
        plane_depth_map = plane_depth_map.clamp(min=0.1, max=max_depth)

        h, w = plane_depth_map.size()[-2:]
        if not plane_depth_map.size() == target_depth.size():
            target_depth = F.interpolate(target_depth.unsqueeze(dim=1), size=(h, w), mode='nearest').squeeze(dim=1)
            mask = F.interpolate(mask.float().unsqueeze(dim=1), size=(h, w), mode='nearest').squeeze(dim=1)
            mask = mask > 0.5

        assert uncertainty is None

        loss = (torch.abs(plane_depth_map[mask] - target_depth[mask])).mean()

        return plane_depth_map, loss

    def make_raft_upsample_mask(self):
        if self.pool:
            up_ratio = 8
        else:
            up_ratio = 4

        self.raft_upsample_mask = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, up_ratio ** 2 * 9, 1, padding=0))

    def make_refine_raft_upsample_mask(self, in_planes, out_planes):
        self.refine_raft_upsample_mask = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes * 9, 1, padding=0))

    def raft_upsample(self, plane_map, mask):
        if self.pool:
            up_ratio = 8
        else:
            up_ratio = 4

        b, _, h, w = plane_map.size()

        mask = mask.view(b, 1, 9, up_ratio, up_ratio, h, w)
        mask = torch.softmax(mask, dim=2)

        up_plane_map = F.unfold(plane_map, [3, 3], padding=1)
        up_plane_map = up_plane_map.view(b, 3, 9, 1, 1, h, w)

        up_plane_map = torch.sum(mask * up_plane_map, dim=2)
        up_plane_map = up_plane_map.permute(0, 1, 4, 2, 5, 3)

        up_plane_map = up_plane_map.reshape(b, 3, up_ratio * h, up_ratio * w)

        return up_plane_map

    def forward(self, ref_feat, src_feat, homo_grids, hypos, ref_img, targets=None, is_test=False):
        # reduce feature channel into half
        ref_feat = self.feat_reduction(ref_feat)
        src_feat = self.feat_reduction(src_feat)

        # avg-pooling to save memory
        if self.pool:
            ref_feat = self.pool_layer(ref_feat)
            src_feat = self.pool_layer(src_feat)

        # warp src feat to target by plane homography
        bs = src_feat.size(0)
        src_coords = self.get_src_coords(bs, homo_grids, self.feat_grids)
        warped_src_feat = self.src_warp(src_coords, src_feat)

        b, c, n, h, w = warped_src_feat.size()

        # concat two view's feature at feature dimension(dim=1)
        if self.feature_type == 'concat':
            feat_volume = torch.cat([warped_src_feat, ref_feat.unsqueeze(dim=2).repeat(1, 1, n, 1, 1)], dim=1)

        elif self.feature_type == 'var':
            ref_feat = ref_feat.unsqueeze(dim=2).repeat(1, 1, n, 1, 1)
            feat_volume_sum = warped_src_feat + ref_feat
            feat_volume_sq_sum = warped_src_feat ** 2 + ref_feat ** 2

            # here we get feat volume variance from the code of Pytorch_mvsnet
            feat_volume = feat_volume_sq_sum.div_(2).sub_(feat_volume_sum.div_(2).pow_(2))

        else:
            raise NotImplementedError

        feat_volume = feat_volume.contiguous()

        # if we use mvsnet network
        conv0 = self.conv0(feat_volume)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        # if do pooling, we do not need conv5
        if self.pool:
            feat_volume = conv4 + self.conv6(conv4)

        else:
            feat_volume = self.conv6(self.conv5(conv4))
            feat_volume = conv4 + self.conv7(feat_volume)

        feat_volume = conv2 + self.conv9(feat_volume)
        feat_volume = conv0 + self.conv11(feat_volume)

        # [b, hypos_num, h, w]
        cost_reg = self.prob(feat_volume).squeeze(dim=1)

        # turn to prob volume
        prob_volume = F.softmax(cost_reg, dim=1)

        # soft-argmin to get plane prediction
        pred_plane_map = self.plane_regression(prob_volume, hypos) #[b,3,h,w]
        pred_plane_map_before_upsample = pred_plane_map

        uncertainty = None

        if self.use_raft_upsample:
            # tune the gradient
            raft_up_mask = 0.25 * self.raft_upsample_mask(ref_feat)
            pred_plane_map = self.raft_upsample(pred_plane_map, raft_up_mask)

        else:
            pred_plane_map = F.interpolate(pred_plane_map, ref_img.size()[-2:], mode='bilinear', align_corners=True)

        if self.apply_refine:
            concat = torch.cat([ref_img, pred_plane_map.type_as(ref_img)], dim=1)

            # estimate residual
            plane_residual = self.refine_conv1(concat)
            plane_residual = self.refine_conv2(plane_residual)
            plane_residual = self.refine_conv3(plane_residual)
            plane_residual = self.refine_res(plane_residual)

            # get refined prediction
            refined_plane_map = pred_plane_map + plane_residual

        if is_test or targets is None:
            if self.apply_refine:
                return pred_plane_map, refined_plane_map, None, None, uncertainty, None

            else:
                return pred_plane_map, None, None, None, uncertainty, None

        else:
            loss = {}
            if not self.finetune_wo_plane_loss and self.with_plane_losses:
                target_plane_map = torch.stack([t.get_field('n_div_d_map') for t in targets], dim=0)
                target_plane_map = target_plane_map.permute(0, 3, 1, 2)

                planar_mask = torch.stack([t.get_field('planar_mask') for t in targets], dim=0)
                # plane loss
                plane_loss = self.plane_l1_loss(pred_plane_map, target_plane_map, planar_mask, uncertainty=uncertainty)

                loss.update(
                    {'loss_stereo_plane': plane_loss}
                )

            # depth loss
            if self.with_pixel_depth_loss:
                planar_depth, planar_depth_loss = self.pixel_depth_loss(pred_plane_map, targets,
                                                                        uncertainty=uncertainty)

                planar_depth_loss = planar_depth_loss * self.cfg.MODEL.STEREO.PLANAR_DEPTH_LOSS_WEIGHT

                loss.update(
                    {'loss_pixel_planar_depth': planar_depth_loss}
                )

            if self.apply_refine:
                if not self.finetune_wo_plane_loss and self.with_plane_losses:
                    # refine plane loss
                    refine_loss = self.plane_l1_loss(refined_plane_map, target_plane_map, planar_mask,
                                                     uncertainty=uncertainty)
                    loss.update(
                        {'loss_stereo_refined_plane': refine_loss})

                # refine depth loss
                if self.with_pixel_depth_loss:
                    refined_planar_depth, refine_planar_depth_loss = self.pixel_depth_loss(refined_plane_map, targets,
                                                                                           uncertainty=uncertainty)

                    # tune the scale of the gradients
                    refine_planar_depth_loss = refine_planar_depth_loss * self.cfg.MODEL.STEREO.PLANAR_DEPTH_LOSS_WEIGHT
                    loss.update(
                        {'loss_pixel_planar_depth_refine': refine_planar_depth_loss}
                    )

            if self.apply_refine:
                return pred_plane_map, refined_plane_map, planar_depth, refined_planar_depth, uncertainty, loss

            else:
                return pred_plane_map, None, planar_depth, None, uncertainty, loss


def build_stereo(cfg):
    stereo_net = StereoNet(cfg)

    return stereo_net
