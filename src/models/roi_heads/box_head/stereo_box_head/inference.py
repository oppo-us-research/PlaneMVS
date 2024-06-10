# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.config import cfg


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

        self.separate_pred = cfg.MODEL.SRCNN.SEPARATE_PRED

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

        if self.separate_pred:
            return self.separate_forward(x, boxes)

        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        num_classes = class_prob.shape[1]

        left_regression = torch.cat([box_regression[:, i*8: (i+1)*8-4] for i in range(num_classes)], dim=-1)
        right_regression = torch.cat([box_regression[:, i*8+4: (i+1)*8] for i in range(num_classes)], dim=-1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box[0].size for box in boxes]
        boxes_per_image = [len(box[0]) for box in boxes]

        left_concat_boxes = torch.cat([a[0].bbox for a in boxes], dim=0)
        right_concat_boxes = torch.cat([a[1].bbox for a in boxes], dim=0)

        left_proposals = self.box_coder.decode(
            left_regression.view(sum(boxes_per_image), -1), left_concat_boxes
        )

        right_proposals = self.box_coder.decode(
            right_regression.view(sum(boxes_per_image), -1), right_concat_boxes
        )

        num_classes = class_prob.shape[1]

        left_proposals = left_proposals.split(boxes_per_image, dim=0)
        right_proposals = right_proposals.split(boxes_per_image, dim=0)

        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for img_idx, (prob, boxes_per_left_img, boxes_per_right_img, image_shape) in enumerate(zip(class_prob, left_proposals, right_proposals, image_shapes)):
            left_boxlist = self.prepare_boxlist(boxes_per_left_img, prob, image_shape)
            left_boxlist = left_boxlist.clip_to_image(remove_empty=False)

            right_boxlist = self.prepare_boxlist(boxes_per_right_img, prob, image_shape)
            right_boxlist = right_boxlist.clip_to_image(remove_empty=False)

            if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                boxlist = self.filter_results(left_boxlist, right_boxlist, num_classes)
            results.append(boxlist)

        return results

    def separate_forward(self, x, boxes):
        (left_class_logits, right_class_logits), (left_box_regression, right_box_regression) = x

        left_class_prob = F.softmax(left_class_logits, -1)
        num_classes = left_class_prob.shape[1]

        right_class_prob = F.softmax(right_class_logits, -1)

        image_shapes = [box[0].size for box in boxes]

        left_boxes_per_image = [len(box[0]) for box in boxes]
        right_boxes_per_image = [len(box[1]) for box in boxes]

        left_class_prob = left_class_prob.split(left_boxes_per_image, dim=0)
        right_class_prob = right_class_prob.split(right_boxes_per_image, dim=0)

        left_concat_boxes = torch.cat([a[0].bbox for a in boxes], dim=0)
        right_concat_boxes = torch.cat([a[1].bbox for a in boxes], dim=0)

        left_proposals = self.box_coder.decode(
            left_box_regression.view(sum(left_boxes_per_image), -1), left_concat_boxes
        )

        right_proposals = self.box_coder.decode(
            right_box_regression.view(sum(right_boxes_per_image), -1), right_concat_boxes
        )

        left_proposals = left_proposals.split(left_boxes_per_image, dim=0)
        right_proposals = right_proposals.split(right_boxes_per_image, dim=0)

        results = []
        for img_idx, (left_prob, right_prob, boxes_per_left_img, boxes_per_right_img, image_shape) in enumerate(zip(left_class_prob, right_class_prob, left_proposals, right_proposals, image_shapes)):
            left_boxlist = self.prepare_boxlist(boxes_per_left_img, left_prob, image_shape)
            left_boxlist = left_boxlist.clip_to_image(remove_empty=False)

            right_boxlist = self.prepare_boxlist(boxes_per_right_img, right_prob, image_shape)
            right_boxlist = right_boxlist.clip_to_image(remove_empty=False)

            if not self.bbox_aug_enabled:
                left_boxlist = self.separate_filter_results(left_boxlist, num_classes)
                right_boxlist = self.separate_filter_results(right_boxlist, num_classes)

            results.append([left_boxlist, right_boxlist])

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

        return boxlist

    def filter_results(self, left_boxlist, right_boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        left_boxes = left_boxlist.bbox.reshape(-1, num_classes * 4)
        right_boxes = right_boxlist.bbox.reshape(-1, num_classes * 4)

        scores = left_boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []

        labels_all = scores.argmax(dim=-1)
        scores_on_label = torch.gather(scores, 1, labels_all.unsqueeze(dim=-1)).squeeze(dim=-1)

        map_for_box = torch.stack([labels_all * 4, labels_all * 4 + 1, labels_all * 4 + 2, labels_all * 4 + 3], dim=-1)

        left_boxes = torch.gather(left_boxes, 1, map_for_box)
        right_boxes = torch.gather(right_boxes, 1, map_for_box)

        keep_bool = (labels_all >= 1) & (scores_on_label > self.score_thresh)

        left_kept_boxlist_all = BoxList(left_boxes[keep_bool, :], left_boxlist.size, mode='xyxy')
        right_kept_boxlist_all = BoxList(right_boxes[keep_bool, :], right_boxlist.size, mode='xyxy')

        left_kept_boxlist_all.add_field('scores', scores_on_label[keep_bool])
        left_kept_boxlist_all.add_field('labels', labels_all[keep_bool])

        right_kept_boxlist_all.add_field('scores', scores_on_label[keep_bool])
        right_kept_boxlist_all.add_field('labels', labels_all[keep_bool])

        _, left_kept_idxs = boxlist_nms(
            left_kept_boxlist_all, self.nms
        )

        _, right_kept_idxs = boxlist_nms(
            right_kept_boxlist_all, self.nms
        )

        left_kept_idxs = list(left_kept_idxs.cpu().numpy())
        right_kept_idxs = list(right_kept_idxs.cpu().numpy())

        intersected_idxs = list(set(left_kept_idxs).intersection(set(right_kept_idxs)))
        intersected_idxs = torch.from_numpy(np.asarray(intersected_idxs)).to(left_boxlist.bbox.device)

        # for some case there is no intersected idxs..
        intersected_idxs = intersected_idxs.long()

        left_kept_boxlist_all = left_kept_boxlist_all[intersected_idxs]
        right_kept_boxlist_all = right_kept_boxlist_all[intersected_idxs]

        result = [left_kept_boxlist_all, right_kept_boxlist_all]

        number_of_detections = len(result[0])
        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result[0].get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)

            left_result = result[0][keep]
            right_result = result[1][keep]

            result = [left_result, right_result]

        return result

    def separate_filter_results(self, boxlist, num_classes):
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field('scores').reshape(-1, num_classes)

        device = scores.device

        labels_all = scores.argmax(dim=-1)
        scores_on_label = torch.gather(scores, 1, labels_all.unsqueeze(dim=-1)).squeeze(dim=-1)

        map_for_box = torch.stack([labels_all * 4, labels_all * 4 + 1, labels_all * 4 + 2, labels_all * 4 + 3], dim=-1)
        boxes = torch.gather(boxes, 1, map_for_box)

        keep_bool = (labels_all >= 1) & (scores_on_label > self.score_thresh)

        kept_boxlist_all = BoxList(boxes[keep_bool, :], boxlist.size, mode='xyxy')

        kept_boxlist_all.add_field('scores', scores_on_label[keep_bool])
        kept_boxlist_all.add_field('labels', labels_all[keep_bool])

        kept_boxlist_all, _ = boxlist_nms(
            kept_boxlist_all, self.nms
        )

        result = kept_boxlist_all

        number_of_detections = len(result)

        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field('scores')
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
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
