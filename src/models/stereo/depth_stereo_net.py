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


class DepthStereoNet(nn.Module):
    def __init__(self, cfg, in_channels=256):
        super(DepthStereoNet, self).__init__()
        self.cfg = cfg

        # concat the features by default
        self.feature_type = cfg.MODEL.STEREO.FEATURE_TYPE

        # reduce the feature channel by 1x1 conv to save memory
        self.feat_reduction = nn.Conv2d(in_channels, 
                                        self.cfg.MODEL.STEREO.FEATURE_REDUCT_CHANNELS, 
                                        1, 1, 0)

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

        # using raft upsampling method(convex combination) 
        # to upsample prediction from 1/8 scale to full size;
        self.use_raft_upsample = cfg.MODEL.STEREO.RAFT_UPSAMPLE

        if self.apply_refine:
            # in_channel_num: 1+3, i.e., depth + rgb channel;
            self.make_refine_net(4, 32)

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

        if self.use_raft_upsample:
            self.make_raft_upsample_mask()

        # stereo feature grids
        self.feat_grids = self.make_grids(self.stereo_w, self.stereo_h)
        # img grids
        self.img_grids = self.make_grids(self.img_w, self.img_h)

        self._init_weights()

    # compute img grids
    def make_grids(self, w, h):
        xxs, yys = np.meshgrid(np.arange(w), np.arange(h))
        xys = np.ones((h, w, 3))
        xys[..., 0] = xxs
        xys[..., 1] = yys

        xys = torch.from_numpy(xys).to('cuda') #[h,w,3]
        xys.requires_grad = False

        return xys

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

        # change plane chan from 3 to 1 for depth
        self.refine_res = nn.Conv2d(out_planes, 1, 3, 1, 1)

    # feature warping
    def src_warp(self, src_coords, src_feat, padding_mode='zeros'):
        b, n, h, w, _ = src_coords.size()
        c = src_feat.size(1)
        src_coords = src_coords.view(b, n * h, w, 2)

        warped_src_feat = F.grid_sample(src_feat, src_coords, mode='bilinear', padding_mode=padding_mode, align_corners=True)
        warped_src_feat = warped_src_feat.view(b, c, n, h, w)

        return warped_src_feat

    # homo_grids: (b, n, 3, 3), src_feat: (b, 32, 120, 160)
    def get_src_coords(self, depth_hypos, targets):
        intrinsic = torch.stack([t.get_field('intrinsic_for_stereo') for t in targets])[..., :3, :3]
        intrinsic = intrinsic.type_as(depth_hypos)

        ref_pose = torch.stack([t.get_field('ref_pose') for t in targets]).type_as(depth_hypos)
        src_pose = torch.stack([t.get_field('src_pose') for t in targets]).type_as(depth_hypos)

        grids = self.feat_grids.permute(2, 0, 1) #[h,w,3]-->[3,h,w]

        depth_hypos = depth_hypos.unsqueeze(dim=1)
        n_depth = depth_hypos.size(-1)

        bs = intrinsic.size(0)
        grids = grids.unsqueeze(0).repeat(bs, 1, n_depth, 1, 1)
        h, w = grids.size()[-2:]

        cam_grids = intrinsic.inverse() @ grids.contiguous().view(bs, 3, -1).type_as(intrinsic)
        cam_grids = cam_grids.view(bs, 3, n_depth, h, w)

        cam_grids = cam_grids * depth_hypos.unsqueeze(-1).unsqueeze(-1)

        homo_ref_points = torch.ones(bs, 4, n_depth, h, w).to(cam_grids.device)
        homo_ref_points[:, :3, ...] = cam_grids

        homo_src_points = src_pose.inverse() @ ref_pose @ homo_ref_points.view(bs, 4, -1)

        src_points = (intrinsic @ homo_src_points[:, :3, ...]).view(bs, -1, n_depth, h, w)
        src_xys = src_points[:, :2, ...] / (src_points[:, -1:, ...] + 1e-10)

        # (b, n, h, w, 2)
        src_xys = src_xys.permute(0, 2, 3, 4, 1)

        workspace = torch.tensor([(w - 1) / 2, (h - 1) / 2]).to(src_xys.device)
        workspace = workspace.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # normalize the src grids
        src_xys = src_xys / workspace - 1

        return src_xys

    # soft-argmax to get plane map prediction
    def depth_regression(self, prob, depth_hypos):
        # (b, n, 120, 160) * (b, n, 1, 1)
        depth_map = torch.sum(prob * depth_hypos.unsqueeze(dim=-1).unsqueeze(dim=-1), dim=1)
        depth_map = depth_map.unsqueeze(1)

        return depth_map

    # transform plane-map to depth-map then compute loss
    def pixel_depth_loss(self, pred_depth_map, targets):
        target_depth = torch.stack([t.get_field('depth') for t in targets], dim=0)
        valid_mask = target_depth > 1e-4

        pred_depth_map = pred_depth_map.squeeze(dim=1)
        loss = torch.mean(torch.abs(pred_depth_map[valid_mask] - target_depth[valid_mask]))

        return loss

    def make_raft_upsample_mask(self):
        if self.pool:
            up_ratio = 8
        else:
            up_ratio = 4

        self.raft_upsample_mask = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, up_ratio**2*9, 1, padding=0))

    def make_refine_raft_upsample_mask(self, in_planes, out_planes):
        self.refine_raft_upsample_mask = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes*9, 1, padding=0))

    def raft_upsample(self, depth_map, mask):
        b, _, h, w = depth_map.size()

        if self.pool:
            up_ratio = 8
        else:
            up_ratio = 4

        mask = mask.view(b, 1, 9, up_ratio, up_ratio, h, w)
        mask = torch.softmax(mask, dim=2)

        up_depth_map = F.unfold(depth_map, [3,3], padding=1)
        up_depth_map = up_depth_map.view(b, 1, 9, 1, 1, h, w)

        up_depth_map = torch.sum(mask * up_depth_map, dim=2)
        up_depth_map = up_depth_map.permute(0, 1, 4, 2, 5, 3)

        up_depth_map = up_depth_map.reshape(b, 1, up_ratio*h, up_ratio*w)

        return up_depth_map
    

    def forward(self, ref_feat, src_feat, hypos, ref_img, targets=None, is_test=False):
        # reduce feature channel into half
        ref_feat = self.feat_reduction(ref_feat)
        src_feat = self.feat_reduction(src_feat)

        # avg-pooling to save memory
        if self.pool:
            ref_feat = self.pool_layer(ref_feat)
            src_feat = self.pool_layer(src_feat)

        # warp src feat to target by plane homography
        src_coords = self.get_src_coords(hypos, targets)
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

        # soft-argmax to get plane prediction
        pred_depth_map = self.depth_regression(prob_volume, hypos)

        if self.use_raft_upsample:
            # tune the gradient
            raft_up_mask = 0.25 * self.raft_upsample_mask(ref_feat)
            # from 1/8 scale upsampled to full scale;
            pred_depth_map = self.raft_upsample(pred_depth_map, raft_up_mask)

        else:
            pred_depth_map = F.interpolate(pred_depth_map, 
                                           ref_img.size()[-2:], mode='bilinear',
                                           align_corners=True)

        if self.apply_refine:
            concat = torch.cat([ref_img, pred_depth_map.type_as(ref_img)], dim=1)

            # estimate residual
            depth_residual = self.refine_conv1(concat)
            depth_residual = self.refine_conv2(depth_residual)
            depth_residual = self.refine_conv3(depth_residual)
            depth_residual = self.refine_res(depth_residual)

            # get refined prediction
            refined_depth_map = pred_depth_map + depth_residual

        if is_test or targets is None:
            if self.apply_refine:
                return pred_depth_map, refined_depth_map

            else:
                return pred_depth_map, None

        else:
            loss = {}
            # depth loss
            if self.with_pixel_depth_loss:
                depth_loss = self.pixel_depth_loss(pred_depth_map, targets)

                loss.update(
                    {'loss_pixel_depth': depth_loss})

            if self.apply_refine:
                # refine depth loss
                if self.with_pixel_depth_loss:
                    refine_planar_depth_loss = self.pixel_depth_loss(refined_depth_map, targets)

                    loss.update(
                        {'loss_pixel_depth_refine': refine_planar_depth_loss}
                    )

            if self.apply_refine:
                return pred_depth_map, refined_depth_map, loss

            else:
                return pred_depth_map, None, loss


def build_depth_stereo(cfg):
    depth_stereo_net = DepthStereoNet(cfg)

    return depth_stereo_net
