# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3


# registry.ROI_MASK_FEATURE_EXTRACTORS.register(
#     "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
# )


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("StereoMaskRCNNFPNFeatureExtractor")
class StereoMaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(StereoMaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        self.separate_pred = cfg.MODEL.SRCNN.SEPARATE_PRED
        if not self.separate_pred:
            input_size = in_channels

        else:
            input_size = in_channels * 2

        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        if not self.separate_pred:
            x_left, x_right = x
            proposals_left = [proposal[0] for proposal in proposals]
            proposals_right = [proposal[1] for proposal in proposals]

            x_left = self.pooler(x_left, proposals_left)
            x_right = self.pooler(x_right, proposals_right)

            for layer_name in self.blocks:
                x_left = F.relu(getattr(self, layer_name)(x_left))
                x_right = F.relu(getattr(self, layer_name)(x_right))

            return [x_left, x_right]

        else:
            x_left, x_right = x
            x_shared = [torch.cat([x_left[i], x_right[i]], dim=1) for i in range(len(x[0]))]

            proposals_left = [proposal[0] for proposal in proposals]
            proposals_right = [proposal[1] for proposal in proposals]

            x_left = self.pooler(x_shared, proposals_left)
            x_right = self.pooler(x_shared, proposals_right)

            for layer_name in self.blocks:
                x_left = F.relu(getattr(self, layer_name)(x_left))
                x_right = F.relu(getattr(self, layer_name)(x_right))

            return [x_left, x_right]


def make_stereo_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
