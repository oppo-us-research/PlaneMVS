"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaneCostModule(nn.Module):
    def __init__(self, cfg):
        super(PlaneCostModule, self).__init__()

        self.cfg = cfg
        self.pdist = nn.PairwiseDistance(p=2)

    def normalize(self, a):
        return (a - a.min()) / (a.max() - a.min() + 1e-8)

    def forward(self, rgb, plane_map):
        bs = rgb.size(0)

        rgb_down = self.pdist(rgb[:, :, 1:], rgb[:, :, :-1])
        rgb_right = self.pdist(rgb[:, :, :, 1:], rgb[:, :, :, :-1])

        rgb_down = torch.stack([self.normalize(rgb_down[i]) for i in range(bs)])
        rgb_right = torch.stack([self.normalize(rgb_right[i]) for i in range(bs)])

        plane_down = self.pdist(plane_map[:, :, 1:], plane_map[:, :, :-1])
        plane_right = self.pdist(plane_map[:, :, :, 1:], plane_map[:, :, :, :-1])

        plane_down = torch.stack([self.normalize(plane_down[i]) for i in range(bs)])
        plane_right = torch.stack([self.normalize(plane_right[i]) for i in range(bs)])

        # [cost_dim, bs, h, w]
        cost_down = torch.stack([rgb_down, plane_down.type_as(rgb_down)])
        cost_right = torch.stack([rgb_right, plane_right.type_as(rgb_right)])

        cost_down, _ = torch.max(cost_down, 0)
        cost_right, _ = torch.max(cost_right, 0)

        cost_map = cost_down[:, :, :-1] + cost_right[:, :-1, :]
        cost_map = F.pad(cost_map, (0, 1, 1, 0), "constant", 0)

        return cost_map


def build_cost(cfg):
    return PlaneCostModule(cfg)
