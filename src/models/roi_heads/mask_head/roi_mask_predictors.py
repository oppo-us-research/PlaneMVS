"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""
# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.modeling import registry


from src import OUR_MODEL_NAME_TAG
_tag = OUR_MODEL_NAME_TAG

@registry.ROI_MASK_PREDICTOR.register(_tag + "MaskRCNNC4Predictor")
class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
    

    # ------------------------------
    # Updated by PlaneMVS's authors;
    # ------------------------------
    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        mask_logits = self.mask_fcn_logits(x)

        uncert_logits = None # add a dummy one;

        return mask_logits, uncert_logits


@registry.ROI_MASK_PREDICTOR.register(_tag+ "MaskRCNNConv1x1Predictor")
class MaskRCNNConv1x1Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNConv1x1Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_inputs = in_channels

        self.mask_fcn_logits = Conv2d(num_inputs, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.mask_fcn_logits(x)


def make_roi_mask_predictor(cfg, in_channels):
    func = registry.ROI_MASK_PREDICTOR[_tag + cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg, in_channels)
