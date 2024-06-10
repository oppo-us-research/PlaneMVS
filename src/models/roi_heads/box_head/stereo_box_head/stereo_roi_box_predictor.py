"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from torch import nn

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.modeling import registry


@registry.ROI_BOX_PREDICTOR.register("StereoFastRCNNPredictor")
class StereoFastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(StereoFastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        raise NotImplementedError

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return x, cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("StereoFPNPredictor")
class StereoFPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(StereoFPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.separate_pred = cfg.MODEL.SRCNN.SEPARATE_PRED

        self.cls_score = nn.Linear(representation_size, num_classes)

        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes

        if not self.separate_pred:
            self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 8)

        else:
            self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if not self.separate_pred:
            if x.ndimension() == 8:
                assert list(x.shape[2:]) == [1, 1]
                x = x.view(x.size(0), -1)
            scores = self.cls_score(x)
            bbox_deltas = self.bbox_pred(x)

            return x, scores, bbox_deltas

        else:
            x_left, x_right = x
            if x_left.ndimension() == 4:
                assert list(x_left.shape[2:]) == [1, 1]
                x_left = x_left.view(x_left.size(0), -1)

            left_scores = self.cls_score(x_left)
            left_bbox_deltas = self.bbox_pred(x_left)

            if x_right.ndimension() == 4:
                assert list(x_right.shape[2:]) == [1, 1]
                x_right = x_right.view(x_right.size(0), -1)

            right_scores = self.cls_score(x_right)
            right_bbox_deltas = self.bbox_pred(x_right)

            return (x_left, x_right), (left_scores, right_scores), (left_bbox_deltas, right_bbox_deltas)


def make_stereo_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
