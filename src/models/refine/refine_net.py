"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
import torch.nn as nn

from .submodules import RefinementBlockMask


class RefinementNet(nn.Module):
    def __init__(self, cfg):
        super(RefinementNet, self).__init__()

        self.cfg = cfg
        self.refinement_block = RefinementBlockMask()
        self.use_soft_mask = cfg.MODEL.REFINE.USE_SOFT_MASK

    def forward(self, image, result):
        if self.use_soft_mask:
            masks = result.get_field('soft_mask').cuda()

        else:
            masks = result.get_field('mask').cuda()

        plane_num = masks.size(0)

        image = image.repeat(plane_num, 1, 1, 1)

        plane_depth = result.get_field('planar_depth').unsqueeze(dim=0).unsqueeze(dim=0).repeat(plane_num, 1, 1, 1).type_as(masks)
        depth = result.get_field('depth').unsqueeze(dim=0).repeat(plane_num, 1, 1, 1).type_as(masks)

        plane_xyz = result.get_field('plane_xyz').type_as(masks)

        masks = torch.cat([masks, plane_xyz], dim=1)

        # only keep the mask channel
        other_masks = (masks.sum(0, keepdim=True) - masks)[:, :1]

        other_inputs = torch.cat([plane_depth, depth, masks, other_masks], dim=1)
        masks = self.refinement_block(image, other_inputs)

        return masks


def build_refine(cfg):
    return RefinementNet(cfg)
