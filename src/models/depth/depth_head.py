"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""
# ------------------------------------------------------------------------------------
# Modified from planercnn (https://github.com/NVlabs/planercnn/blob/master/models/model.py)
# Copyright (c) 2017 Matterport, Inc.
# ------------------------------------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn

# Modified based on the code:
# https://github.com/NVlabs/planercnn/blob/2698414a44eaa164f5174f7fe3c87dfc4d5dea3b/models/model.py#L1090
class DepthHead(nn.Module):
    def __init__(self,
                 cfg,
                 upsample_method='bilinear',
                 upsample_ratio=2):
        super(DepthHead, self).__init__()

        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio

        self.loss_depth = cfg.MODEL.DEPTH.LOSS_TYPE

        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.deconv1 = nn.Sequential(
            nn.Upsample(size=(15, 20), mode='nearest', align_corners=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.deconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.deconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.deconv5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest', align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.depth_head = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        if self.upsample_method is None:
            self.upsample = None

        elif self.upsample_method == 'deconv':
            raise NotImplementedError

        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method, align_corners=True)

    def forward(self, feats, targets=None):
        feats = feats[::-1]

        x = self.deconv1(self.conv1(feats[0]))
        x = self.deconv2(torch.cat([self.conv2(feats[1]), x], dim=1))
        x = self.deconv3(torch.cat([self.conv3(feats[2]), x], dim=1))
        x = self.deconv4(torch.cat([self.conv4(feats[3]), x], dim=1))
        feats = self.deconv5(torch.cat([self.conv5(feats[4]), x], dim=1))

        x = self.depth_head(feats)

        if self.upsample is not None:
            x = self.upsample(x).squeeze(dim=1)

        if self.training:
            assert targets is not None

            target = torch.stack([target.get_field('depth') for target in targets])
            assert x.size() == target.size()

            if self.loss_depth == 'L1Loss':
                valid_mask = target > 1e-4
                loss = torch.mean(torch.abs(x[valid_mask] - target[valid_mask]))

            else:
                raise NotImplementedError

            loss = {
                'loss_depth': loss
            }

            return x, feats, loss

        else:
            return x, feats, {}


def build_depth(cfg):
    return DepthHead(cfg)
