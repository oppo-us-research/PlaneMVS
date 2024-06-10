"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder

""" load our own modules """
from src.config import cfg
from src.structures.boxlist_ops import boxlist_nms


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled

        self.nms_per_cls = cfg.MODEL.ROI_HEADS.NMS_PER_CLS

    def normal_decode(self, normal_cls_prob, normal_res_regression, anchor_normals):
        normal_cls_pred = normal_cls_prob.argmax(dim=-1)
        map_inds = torch.stack([normal_cls_pred * 3, normal_cls_pred * 3 + 1, normal_cls_pred * 3 + 2], dim=-1)
        normal_residual = torch.gather(normal_res_regression, 1, map_inds)

        if isinstance(anchor_normals, np.ndarray):
            anchor_normals = torch.from_numpy(anchor_normals)
        anchor_normals = anchor_normals.to(normal_cls_pred.device)

        selected_anchors = anchor_normals[normal_cls_pred]
        normal_res = selected_anchors + normal_residual

        return normal_cls_pred, normal_res

    def forward(self, x, boxes, normal_preds=None, anchor_normals=None):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        if normal_preds is not None:
            assert anchor_normals is not None

            normal_logits, normal_res_regression = normal_preds
            normal_prob = F.softmax(normal_logits, -1)
            normal_cls, normal_reg = self.normal_decode(normal_prob, normal_res_regression, anchor_normals)

            normal_cls = normal_cls.split(boxes_per_image, dim=0)
            normal_reg = normal_reg.split(boxes_per_image, dim=0)

        results = []
        for img_idx, (prob, boxes_per_img, image_shape) in enumerate(zip(class_prob, proposals, image_shapes)):
            if normal_preds is not None:
                boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape,
                                               normal_cls[img_idx], normal_reg[img_idx])

            else:
                boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)

            boxlist = boxlist.clip_to_image(remove_empty=False)
            if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape, normal_cls=None, normal_reg=None):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)

        if normal_cls is not None:
            boxlist.add_field("normal_cls", normal_cls)

        if normal_reg is not None:
            boxlist.add_field("normal_res", normal_reg)

        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []

        if self.nms_per_cls:
            # Apply threshold on detection probabilities and apply NMS
            # Skip j = 0, because it's the background class
            inds_all = scores > self.score_thresh
            for j in range(1, num_classes):
                inds = inds_all[:, j].nonzero().squeeze(1)
                scores_j = scores[inds, j]
                boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                boxlist_for_class, _ = boxlist_nms(
                    boxlist_for_class, self.nms
                )
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
                )
                result.append(boxlist_for_class)

            result = cat_boxlist(result)

        else:
            labels_all = scores.argmax(dim=-1)
            scores_on_label = torch.gather(scores, 1, labels_all.unsqueeze(dim=-1)).squeeze(dim=-1)

            map_for_box = torch.stack([labels_all * 4, labels_all * 4 + 1, labels_all * 4 + 2, labels_all * 4 + 3], dim=-1)
            boxes = torch.gather(boxes, 1, map_for_box)

            keep_bool = (labels_all >= 1) & (scores_on_label > self.score_thresh)

            kept_boxlist_all = BoxList(boxes[keep_bool, :], boxlist.size, mode='xyxy')

            kept_boxlist_all.add_field('scores', scores_on_label[keep_bool])
            kept_boxlist_all.add_field('labels', labels_all[keep_bool])

            if 'normal_cls' in boxlist.fields():
                kept_boxlist_all.add_field('normal_cls', boxlist.get_field('normal_cls')[keep_bool])

            if 'normal_res' in boxlist.fields():
                kept_boxlist_all.add_field('normal_res', boxlist.get_field('normal_res')[keep_bool])

            kept_boxlist_all, _ = boxlist_nms(
                kept_boxlist_all, self.nms
            )

            result = kept_boxlist_all

        number_of_detections = len(result)
        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple = False).squeeze(1)
            result = result[keep]

        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled
    )
    return postprocessor
