"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

from torch import nn


class NormalPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(NormalPredictor, self).__init__()

        assert in_channels is not None
        self.cfg = config

        self.normal_cls = nn.Linear(in_channels, self.cfg.MODEL.ANCHOR_NORMAL_NUM)
        self.normal_res = nn.Linear(in_channels, self.cfg.MODEL.ANCHOR_NORMAL_NUM * 3)

        nn.init.normal_(self.normal_cls.weight, mean=0, std=0.01)
        nn.init.constant_(self.normal_cls.bias, 0)

        nn.init.normal_(self.normal_res.weight, mean=0, std=0.01)
        nn.init.constant_(self.normal_cls.bias, 0)

    def forward(self, x):
        normal_cls_pred = self.normal_cls(x)
        normal_res_pred = self.normal_res(x)

        return normal_cls_pred, normal_res_pred


def make_normal_predictor(cfg, in_channels):
    func = NormalPredictor
    return func(cfg, in_channels)
