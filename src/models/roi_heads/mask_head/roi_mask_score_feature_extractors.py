"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
from torch import nn
from torch.nn import functional as F

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.layers import Conv2d


class MaskScoreFeatureExtractor(nn.Module):
    """
        MaskScore head feature extractor.
    """

    def __init__(self, cfg, in_channels):
        super(MaskScoreFeatureExtractor, self).__init__()

        self.maskscore_fcn1 = Conv2d(in_channels, 256, 3, 1, 1)
        self.maskscore_fcn2 = Conv2d(256, 256, 3, 1, 1)
        self.maskscore_fcn3 = Conv2d(256, 256, 3, 1, 1)

        self.maskscore_fcn4 = Conv2d(256, 256, 3, 2, 1)
        self.maskscore_fc1 = nn.Linear(256*7*7, 1024)
        self.maskscore_fc2 = nn.Linear(1024, 1024)

        self.out_channels = 1024

        for l in [self.maskscore_fcn1, self.maskscore_fcn2, self.maskscore_fcn3, self.maskscore_fcn4]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        for l in [self.maskscore_fc1, self.maskscore_fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)


    def forward(self, x, mask):
        mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = torch.cat((x, mask_pool), 1)
        x = F.relu(self.maskscore_fcn1(x))
        x = F.relu(self.maskscore_fcn2(x))
        x = F.relu(self.maskscore_fcn3(x))
        x = F.relu(self.maskscore_fcn4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.maskscore_fc1(x))
        x = F.relu(self.maskscore_fc2(x))

        return x

def make_roi_mask_score_feature_extractor(cfg, in_channels):
    return MaskScoreFeatureExtractor(cfg, in_channels)
