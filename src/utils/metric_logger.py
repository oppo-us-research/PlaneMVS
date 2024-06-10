"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

from collections import defaultdict
from collections import deque

import time
from datetime import datetime

import torch

import numpy as np

from third_party.maskrcnn_main.maskrcnn_benchmark.utils.metric_logger import SmoothedValue
from third_party.maskrcnn_main.maskrcnn_benchmark.utils.comm import is_main_process


# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# Copyright (c) 2024 OPPO. All rights reserved.
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    # ------------------------------
    # Updated by PlaneMVS's authors;
    # ------------------------------
    def update(self, 
              iteration=None, # added a dummy arg;
              **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f"Got (k,v)={k}, {type(v)}"
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

# ------------------------------
# Added by PlaneMVS's authors;
# ------------------------------
class TensorboardLogger(MetricLogger):
    def __init__(self,
                 log_dir,
                 delimiter='\t'):

        super(TensorboardLogger, self).__init__(delimiter)
        self.writer = self._get_tensorboard_writer(log_dir)

    @staticmethod
    def _get_tensorboard_writer(log_dir):
        try:
            #from tensorboardX import SummaryWriter
            from torch.utils.tensorboard import SummaryWriter
        except:
            raise ImportError(
                "To use tensorboard please intall tensorboardX"
                "[ pip install tensorboard tensorboardX  ]."
            )

        if is_main_process():
            timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H:%M")
            tb_logger = SummaryWriter('{}-{}'.format(log_dir, timestamp))
            return tb_logger

        else:
            return None

    # update train metric
    def update(self, iteration=None, **kwargs):
        super(TensorboardLogger, self).update(iteration=iteration, **kwargs)
        if self.writer:
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                assert isinstance(v, (float, int))

                if not np.isnan(v):
                    self.writer.add_scalar(k, v, iteration)
    
    # update validation metric
    def update_val(self, iteration, meters_val, images_dict = None):
        if self.writer:
            for k, v in meters_val.meters.items():
                if not np.isnan(v.global_avg):
                    print(k, v.global_avg)
                    self.writer.add_scalar('val_' + k, v.global_avg, iteration)
            
            # added by PlaneMVS's authors;
            # if images_dict is not None:
            #     images_dict = tensor2numpy(images_dict)
            #     for k, v in images_dict.items():
            #                 logger.add_image(image_name, 
            #         vutils.make_grid(value, padding=0, nrow=1, normalize=True, scale_each=True),
            #         global_step
            #     )    
