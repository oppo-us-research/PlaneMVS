# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from torch import nn
from maskrcnn_benchmark.layers.misc import interpolate

from maskrcnn_benchmark.structures.bounding_box import BoxList


# TODO check if want to return a single BoxList or a composite
# object
class MaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, cfg, masker=None, soft_masker=None):
        super(MaskPostProcessor, self).__init__()
        self.masker = masker

        self.apply_mask_nms = cfg.MODEL.ROI_HEADS.APPLY_MASK_NMS
        self.mask_nms_thresh = cfg.MODEL.ROI_HEADS.MASK_NMS_THRESH

        self.soft_masker = soft_masker

        self.separate_pred = cfg.MODEL.SRCNN.SEPARATE_PRED

    def mask_iou(self, mask, mask_set):
        i = (mask * mask_set).sum(-1).sum(-1)
        u = mask.sum(-1).sum(-1)

        iou = i / (u + 1e-4)

        return iou

    def mask_nms(self, result):
        if result.bbox.size(0) <= 1:
            return None

        rank = result.get_field('scores').argsort(descending=True)
        result = result[rank]

        masks = result.get_field('mask').squeeze(dim=1)

        keeps = []
        kept_idxs = torch.arange(masks.size(0))

        while len(masks) > 1:
            keeps.append(kept_idxs[0])
            keep = self.mask_iou(masks[0:1], masks[1:]) < self.mask_nms_thresh

            kept_idxs = kept_idxs[1:][keep]
            masks = masks[1:][keep]

            if len(masks) == 1:
                keeps.append(kept_idxs[0])
                break

        keeps = torch.stack(keeps)

        return keeps

    def forward(self, x, boxes, filter_empty=True):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        if self.separate_pred:
            return self.separate_forward(x, boxes, filter_empty=filter_empty)

        left_mask_logits, right_mask_logits = x
        left_mask_prob = left_mask_logits.sigmoid()
        right_mask_prob = right_mask_logits.sigmoid()

        # select masks coresponding to the predicted classes
        num_masks = x[0].shape[0]
        labels = [bbox[0].get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)

        left_mask_prob = left_mask_prob[index, labels][:, None]
        right_mask_prob = right_mask_prob[index, labels][:, None]

        boxes_per_image = [len(box[0]) for box in boxes]

        left_mask_prob = left_mask_prob.split(boxes_per_image, dim=0)
        right_mask_prob = right_mask_prob.split(boxes_per_image, dim=0)

        if self.masker:
            left_boxes = [box[0] for box in boxes]
            left_mask_prob = self.masker(left_mask_prob, left_boxes)

            right_boxes = [box[1] for box in boxes]
            right_mask_prob = self.masker(right_mask_prob, right_boxes)

        results = []
        for idx, (left_prob, right_prob, left_box, right_box) in enumerate(zip(left_mask_prob, right_mask_prob, left_boxes, right_boxes)):
            left_bbox = BoxList(left_box.bbox, left_box.size, mode="xyxy")
            for field in left_box.fields():
                left_bbox.add_field(field, left_box.get_field(field))
            left_bbox.add_field("mask", left_prob)

            right_bbox = BoxList(right_box.bbox, right_box.size, mode="xyxy")
            for field in right_box.fields():
                right_bbox.add_field(field, right_box.get_field(field))
            right_bbox.add_field("mask", right_prob)

            if filter_empty:
                left_valid_idxs = left_prob.squeeze(1).sum(-1).sum(-1) > 0
                right_valid_idxs = right_prob.squeeze(1).sum(-1).sum(-1) > 0

                valid_idxs = left_valid_idxs * right_valid_idxs

                left_bbox = left_bbox[valid_idxs]
                right_bbox = right_bbox[valid_idxs]

            results.append([left_bbox, right_bbox])

        if self.apply_mask_nms:
            new_results = []
            for result in results:
                left_result, right_result = result
                left_keep_idxs = self.mask_nms(left_result)
                right_keep_idxs = self.mask_nms(right_result)

                if left_keep_idxs is None or right_keep_idxs is None:
                    new_results.append(result)
                    continue

                left_keep_idxs = list(left_keep_idxs.cpu().numpy())
                right_keep_idxs = list(right_keep_idxs.cpu().numpy())

                intersected_idxs = list(set(left_keep_idxs).intersection(right_keep_idxs))
                intersected_idxs = torch.from_numpy(np.asarray(intersected_idxs)).to(left_result.bbox.device)

                left_result = left_result[intersected_idxs]
                right_result = right_result[intersected_idxs]

                new_results.append([left_result, right_result])

            results = new_results

        merged_results = []
        # re-write the boxlist format
        for result in results:
            right_bbox = result[1].bbox
            right_mask = result[1].get_field('mask')

            result[0].add_field('src_bbox', right_bbox)
            result[0].add_field('src_mask', right_mask)

            merged_results.append(result[0])

        results = merged_results

        return results

    def separate_forward(self, x, boxes, filter_empty=True):
        left_mask_logits, right_mask_logits = x
        left_mask_prob = left_mask_logits.sigmoid()
        right_mask_prob = right_mask_logits.sigmoid()

        left_num_masks = x[0].shape[0]
        left_labels = [bbox[0].get_field('labels') for bbox in boxes]
        left_labels = torch.cat(left_labels)
        left_index = torch.arange(left_num_masks, device=left_labels.device)

        left_mask_prob = left_mask_prob[left_index, left_labels][:, None]
        left_boxes_per_image = [len(box[0]) for box in boxes]
        left_mask_prob = left_mask_prob.split(left_boxes_per_image, dim=0)

        right_num_masks = x[1].shape[0]
        right_labels = [bbox[1].get_field('labels') for bbox in boxes]
        right_labels = torch.cat(right_labels)
        right_index = torch.arange(right_num_masks, device=right_labels.device)

        right_mask_prob = right_mask_prob[right_index, right_labels][:, None]
        right_boxes_per_image = [len(box[1]) for box in boxes]
        right_mask_prob = right_mask_prob.split(right_boxes_per_image, dim=0)

        if self.masker:
            left_boxes = [box[0] for box in boxes]
            left_mask_prob = self.masker(left_mask_prob, left_boxes)

            right_boxes = [box[1] for box in boxes]
            right_mask_prob = self.masker(right_mask_prob, right_boxes)

        results = []
        for idx, (left_prob, right_prob, left_box, right_box) in enumerate(zip(left_mask_prob, right_mask_prob, left_boxes, right_boxes)):
            left_bbox = BoxList(left_box.bbox, left_box.size, mode='xyxy')
            for field in left_box.fields():
                left_bbox.add_field(field, left_box.get_field(field))
            left_bbox.add_field('mask', left_prob)

            right_bbox = BoxList(right_box.bbox, right_box.size, mode='xyxy')
            for field in right_box.fields():
                right_bbox.add_field(field, right_box.get_field(field))
            right_bbox.add_field('mask', right_prob)

            if filter_empty:
                left_valid_idxs = left_prob.squeeze(1).sum(-1).sum(-1) > 0
                right_valid_idxs = right_prob.squeeze(1).sum(-1).sum(-1) > 0

                left_bbox = left_bbox[left_valid_idxs]
                right_bbox = right_bbox[right_valid_idxs]

            results.append([left_bbox, right_bbox])

        if self.apply_mask_nms:
            new_results = []
            for result in results:
                left_result, right_result = result
                left_keep_idxs = self.mask_nms(left_result)
                right_keep_idxs = self.mask_nms(right_result)

                if left_keep_idxs is not None:
                    left_result = left_result[left_keep_idxs]

                if right_keep_idxs is not None:
                    right_result = right_result[right_keep_idxs]

                new_results.append([left_result, right_result])

            results = new_results

        merged_results = []
        for result in results:
            right_score = result[1].get_field('scores')
            right_bbox = result[1].bbox
            right_mask = result[1].get_field('mask')
            right_labels = result[1].get_field('labels')

            result[0].add_field('src_scores', right_score)
            result[0].add_field('src_bbox', right_bbox)
            result[0].add_field('src_mask', right_mask)
            result[0].add_field('src_labels', right_labels)

            merged_results.append(result[0])

        results = merged_results

        return results


class MaskPostProcessorCOCOFormat(MaskPostProcessor):
    """
    From the results of the CNN, post process the results
    so that the masks are pasted in the image, and
    additionally convert the results to COCO format.
    """

    def forward(self, x, boxes):
        import pycocotools.mask as mask_util
        import numpy as np

        results = super(MaskPostProcessorCOCOFormat, self).forward(x, boxes)
        for result in results:
            masks = result.get_field("mask").cpu()
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")
            result.add_field("mask", rles)
        return results


# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily gor paste_mask_in_image
def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))

    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    # Need to work on the CPU, where fp16 isn't supported - cast to float to avoid this
    mask = mask.float()
    box = box.float()

    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)

    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = mask.to(torch.float32)
    mask = interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    if thresh >= 0:
        mask = mask > thresh

    # else:
        # for visualization and debugging, we also
        # allow it to return an unmodified mask
        # mask = (mask * 255).to(torch.bool)

    if thresh >= 0:
        mask_type = torch.bool
    else:
        mask_type = torch.float32

    im_mask = torch.zeros((im_h, im_w), dtype=mask_type)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    return im_mask


class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert("xyxy")
        im_w, im_h = boxes.size
        res = [
            paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
            for mask, box in zip(masks, boxes.bbox)
        ]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]

        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."

        # TODO:  Is this JIT compatible?
        # If not we should make it compatible.
        results = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), "Number of objects should be the same."
            result = self.forward_single_image(mask, box)
            results.append(result)
        return results


def make_roi_mask_post_processor(cfg):
    if cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS:
        mask_threshold = cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
        masker = Masker(threshold=mask_threshold, padding=1)
    else:
        masker = None

    if cfg.MODEL.STEREO.WITH_PIXEL_INSTANCE_CONSISTENCY_LOSS:
        soft_masker = Masker(threshold=-1, padding=1)
    else:
        soft_masker = None

    mask_post_processor = MaskPostProcessor(cfg, masker, soft_masker)
    return mask_post_processor
