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
import torch.nn.functional as F
from torch import nn

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from maskrcnn_benchmark.modeling.rpn.inference import make_rpn_postprocessor
from maskrcnn_benchmark.modeling.rpn.anchor_generator import make_anchor_generator

""" load our own modules """
from .stereo_loss import make_stereo_rpn_loss_evaluator
from src import OUR_MODEL_NAME_TAG
# tag is used to avoid duplicate names for module_name in registry;
_tag = OUR_MODEL_NAME_TAG

@registry.RPN_HEADS.register(_tag + "SingleConvStereoRPNHead")
class StereoRPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(StereoRPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        # since the feature map is shared at rpn, we must predict two scores at the same head
        # or we can initialize two heads, but that will cost more memory and make it non-balanced
        if cfg.MODEL.SRCNN.SEPARATE_PRED:
            self.cls_logits = nn.Conv2d(2 * in_channels, 2 * num_anchors, kernel_size=1, stride=1)

        else:
            self.cls_logits = nn.Conv2d(2 * in_channels, num_anchors, kernel_size=1, stride=1)

        # here we have 8 position logits to pred in stereo case
        self.bbox_pred = nn.Conv2d(
            2 * in_channels, num_anchors * 8, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x_left, x_right):
        logits = []
        bbox_reg = []

        # for each feature level
        for i in range(len(x_left)):
            t = torch.cat([F.relu(self.conv(x_left[i])),
                          F.relu(self.conv(x_right[i]))], dim=1)

            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))

        return logits, bbox_reg


class StereoRPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and outputs
    RPN proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        super(StereoRPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True, for_stereo=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False, for_stereo=True)

        loss_evaluator = make_stereo_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features_left, features_right, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression = self.head(features_left, features_right)

        features = []
        for i in range(len(features_left)):
            features.append(torch.cat([features_left[i], features_right[i]], dim=1))

        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )

        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_stereo_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)

    return StereoRPNModule(cfg, in_channels)
