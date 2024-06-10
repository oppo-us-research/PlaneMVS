"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from torch import nn

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.layers.misc import interpolate
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import expand_boxes, expand_masks


""" load our own modules """
from src.tools.utils import print_indebugmode


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

        # ------------------------------
        # Added by PlaneMVS's authors;
        # ------------------------------
        self.apply_mask_nms = cfg.MODEL.ROI_HEADS.APPLY_MASK_NMS
        self.mask_nms_thresh = cfg.MODEL.ROI_HEADS.MASK_NMS_THRESH
        self.soft_masker = soft_masker
        self.with_mask_score_head = cfg.MODEL.ROI_MASK_SCORE_HEAD.ACTIVATE
        self.with_pseudo_gt = cfg.MODEL.REFINE.GENERATE_PSEUDO_GT

    # ------------------------------
    # Added by PlaneMVS's authors;
    # ------------------------------
    def mask_iou(self, mask, mask_set):
        i = (mask * mask_set).sum(-1).sum(-1)
        u = mask.sum(-1).sum(-1)

        iou = i / (u + 1e-4)

        return iou

    # ------------------------------
    # Added by PlaneMVS's authors;
    # ------------------------------
    def mask_nms(self, results):
        new_results = []

        if self.with_mask_score_head:
            score_field = 'final_scores'

        else:
            score_field = 'scores'

        for result in results:
            if result.bbox.size(0) <= 1:
                new_results.append(result)
                continue

            rank = result.get_field(score_field).argsort(descending=True)
            #print_indebugmode(f"rank = {rank.device}, result = {result}")
            result = result[rank]

            masks = result.get_field('mask').squeeze(dim=1)

            keeps = []
            kept_idxs = torch.arange(masks.size(0), device=masks.device)

            while len(masks) > 1:
                keeps.append(kept_idxs[0])
                keep = self.mask_iou(masks[0:1], masks[1:]) < self.mask_nms_thresh

                kept_idxs = kept_idxs[1:][keep]
                masks = masks[1:][keep]

                if len(masks) == 1:
                    keeps.append(kept_idxs[0])
                    break

            keeps = torch.stack(keeps)
            result = result[keeps]
            new_results.append(result)

        return new_results

    def forward(self, x, boxes, mask_score_logits=None, filter_empty=True):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_prob = x.sigmoid()
        #print_indebugmode(f"??? x  = {x.device}, {x.shape}")

        # select masks coresponding to the predicted classes
        num_masks = x.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels, dim=0) # added dim=0;
        #print_indebugmode(f"??? labels  = {labels.device}, {labels.shape}")

        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]
        #print_indebugmode(f"??? mask_prob  = {mask_prob.device}, {mask_prob.shape}")

        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)
        #print_indebugmode (f"??? boxes_per_image = {boxes_per_image}, mask_prob len = {len(mask_prob)} , {mask_prob[0].shape}, {mask_prob[0].device}")
        #print_indebugmode (f"??? boxes len = {len(boxes)}, {boxes[0].device}")

        # scatter mask logits into list
        mask_logits = x.split(boxes_per_image, dim=0)

        if self.soft_masker:
            #print_indebugmode ("do self.soft_masker ???")
            soft_mask_prob = self.soft_masker(mask_prob, boxes)

        if self.masker:
            #print_indebugmode ("do self.masker ???")
            mask_prob = self.masker(mask_prob, boxes)

        if mask_score_logits is not None:
            mask_score_logits = mask_score_logits.split(boxes_per_image, dim=0)

        else:
            mask_score_logits = None

        results = []
        for idx, (prob, box) in enumerate(zip(mask_prob, boxes)):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            #print_indebugmode (f"??? add mask: prob device = {prob.device}")
            bbox.add_field("mask", prob)

            if self.with_pseudo_gt:
                img_mask_logits = mask_logits[idx]
                bbox.add_field('mask_logits', img_mask_logits)

            if mask_score_logits is not None:
                cls_scores = bbox.get_field('scores')
                bbox.add_field('plane_scores', mask_score_logits[idx])
                bbox.add_field('final_scores', cls_scores * mask_score_logits[idx])

            if self.soft_masker:
                soft_prob = soft_mask_prob[idx]
                bbox.add_field("soft_mask", soft_prob)

            if filter_empty:
                valid_idxs = prob.squeeze(1).sum(-1).sum(-1) > 0
                bbox = bbox[valid_idxs]

            results.append(bbox)

        if self.apply_mask_nms:
            results = self.mask_nms(results)

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

    im_mask = torch.zeros((im_h, im_w), dtype=mask_type, device=mask.device)
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
            #print_indebugmode (f"??? res 0 = {res.device}")
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
            #print_indebugmode (f"??? res 1 = {res.device}")
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

    if cfg.MODEL.REFINE.USE_SOFT_MASK or cfg.MODEL.STEREO.SOFT_POOL:
        soft_masker = Masker(threshold=-1, padding=1)
    else:
        soft_masker = None

    mask_post_processor = MaskPostProcessor(cfg, masker, soft_masker)
    return mask_post_processor
