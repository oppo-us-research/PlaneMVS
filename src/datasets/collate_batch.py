"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""
# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------------

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.structures.image_list import to_image_list


""" load our own modules """
from src.config import cfg


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))

        images = transposed_batch[0]

        if cfg.MODEL.METHOD == 'single':
            for image in images:
                image['ref_img'] = to_image_list(image['ref_img'], self.size_divisible)

        elif cfg.MODEL.METHOD == 'stereo' \
                or cfg.MODEL.METHOD == 'refine' \
                or cfg.MODEL.METHOD == 'srcnn' \
                or cfg.MODEL.METHOD == 'consistency':
            for image in images:
                image['src_img'] = to_image_list(image['src_img'], self.size_divisible)
                image['ref_img'] = to_image_list(image['ref_img'], self.size_divisible)

                if cfg.MODEL.METHOD == 'consistency' or cfg.MODEL.STEREO.WITH_DETECTION_CONSISTENCY:
                    image['masked_img'] = to_image_list(image['masked_img'], self.size_divisible)

        # we do the image_list convertion in the model class
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]

        return images, targets, img_ids


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))
