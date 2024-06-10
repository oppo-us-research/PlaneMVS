"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
import torch.nn as nn


class ConvBlock(torch.nn.Module):
    """The block consists of a convolution layer, an optional batch normalization layer, and a ReLU layer"""

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, output_padding=0, mode='conv', use_bn=True):
        super(ConvBlock, self).__init__()

        self.use_bn = use_bn

        if mode == 'conv':
            self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=not self.use_bn)

        elif mode == 'deconv':
            self.conv = torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=not self.use_bn)

        elif mode == 'upsample':
            self.conv = torch.nn.Sequential(torch.nn.Upsample(scale_factor=stride, mode='nearest', align_corners=False), torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=not self.use_bn))

        else:
            raise NotImplementedError

        if '3d' not in mode:
            self.bn = torch.nn.BatchNorm2d(out_planes)
        else:
            self.bn = torch.nn.BatchNorm3d(out_planes)

        self.relu = torch.nn.ReLU(inplace=True)

        return

    def forward(self, inp):
        if self.use_bn:
            return self.relu(self.bn(self.conv(inp)))
        else:
            return self.relu(self.conv(inp))


class RefinementBlockMask(torch.nn.Module):
    def __init__(self):
        super(RefinementBlockMask, self).__init__()

        use_bn = False

        self.conv_0 = ConvBlock(3 + 5 + 2, 32, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
        self.conv_1 = ConvBlock(64, 64, kernel_size=3, stride=2, padding=1, use_bn=use_bn)

        self.conv_1_1 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
        self.conv_2 = ConvBlock(128, 128, kernel_size=3, stride=2, padding=1, use_bn=use_bn)
        self.conv_2_1 = ConvBlock(256, 128, kernel_size=3, stride=1, padding=1, use_bn=use_bn)

        self.up_2 = ConvBlock(128, 64, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)
        self.up_1 = ConvBlock(128, 32, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)
        self.pred = nn.Sequential(ConvBlock(64, 16, kernel_size=3, stride=1, padding=1, mode='conv', use_bn=use_bn),
                                  torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))

        self.global_up_2 = ConvBlock(128, 64, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)
        self.global_up_1 = ConvBlock(128, 32, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)

        self.global_pred = nn.Sequential(ConvBlock(64, 16, kernel_size=3, stride=1, padding=1, mode='conv', use_bn=use_bn),
                                         torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))

    def accumulate(self, x):
        return torch.cat([x, (x.sum(0, keepdim=True) - x) / max(len(x) - 1, 1)], dim=1)

    def forward(self, image, other_inputs):
        x_0 = torch.cat([image, other_inputs], dim=1)

        x_0 = self.conv_0(x_0)
        x_1 = self.conv_1(self.accumulate(x_0))
        x_1 = self.conv_1_1(self.accumulate(x_1))
        x_2 = self.conv_2(self.accumulate(x_1))
        x_2 = self.conv_2_1(self.accumulate(x_2))

        y_2 = self.up_2(x_2)
        y_1 = self.up_1(torch.cat([y_2, x_1], dim=1))
        y_0 = self.pred(torch.cat([y_1, x_0], dim=1))

        global_y_2 = self.global_up_2(x_2.mean(dim=0, keepdim=True))
        global_y_1 = self.global_up_1(torch.cat([global_y_2, x_1.mean(dim=0, keepdim=True)], dim=1))
        global_mask = self.global_pred(torch.cat([global_y_1, x_0.mean(dim=0, keepdim=True)], dim=1))

        y_0 = torch.cat([global_mask[:, 0], y_0.squeeze(1)], dim=0)

        return y_0
