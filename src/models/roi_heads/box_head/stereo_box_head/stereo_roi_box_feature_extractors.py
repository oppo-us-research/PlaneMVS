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
from torch import nn
from torch.nn import functional as F

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("StereoResNet50Conv5ROIFeatureExtractor")
class StereoResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

        raise NotImplementedError

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("StereoFPN2MLPFeatureExtractor")
class StereoFPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(StereoFPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        # feature concat
        input_size = 2 * in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.separate_pred = cfg.MODEL.SRCNN.SEPARATE_PRED

    def forward(self, x, proposals):
        if not self.separate_pred:
            x_left, x_right = x
            proposals_left = [proposal[0] for proposal in proposals]
            proposals_right = [proposal[1] for proposal in proposals]

            x_left = self.pooler(x_left, proposals_left)
            x_right = self.pooler(x_right, proposals_right)

            # the features are shared for stereo case
            x = torch.cat([x_left, x_right], dim=1)
            x = x.view(x.size(0), -1)

            x = F.relu(self.fc6(x))
            x = F.relu(self.fc7(x))

            return x

        else:
            x_left, x_right = x
            x_shared = [torch.cat([x_left[i], x_right[i]], dim=1) for i in range(len(x[0]))]

            proposals_left = [proposal[0] for proposal in proposals]
            proposals_right = [proposal[1] for proposal in proposals]

            x_left = self.pooler(x_shared, proposals_left)
            x_right = self.pooler(x_shared, proposals_right)

            x_left = x_left.view(x_left.size(0), -1)
            x_left = F.relu(self.fc6(x_left))
            x_left = F.relu(self.fc7(x_left))

            x_right = x_right.view(x_right.size(0), -1)
            x_right = F.relu(self.fc6(x_right))
            x_right = F.relu(self.fc7(x_right))

            return [x_left, x_right]


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("StereoFPNXconv1fcFeatureExtractor")
class StereoFPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(StereoFPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

        raise NotImplementedError

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_stereo_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        _tag + 
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
