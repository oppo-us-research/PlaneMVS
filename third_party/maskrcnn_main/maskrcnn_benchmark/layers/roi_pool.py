# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# Copyright (c) 2024 OPPO. All rights reserved.
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

# ------------------------------
""" Updated by PlaneMVS's authors; """
# Since we just build libs in the local `build`` dir,
# We do not install them in the system-level, like /usr/local/* 
# via something like `pip install maskrcnn_benchmark`;
# We explicitly load it from the local `build` dir in this project, 
# once the library was compiled. 
# You can run `./compile.sh` to compile the cuda codes 
# (including: maskrcnn_benchmark/csrc/cuda/*.cu);
from build.lib.maskrcnn_benchmark import _C
# ------------------------------

# ------------------------------
""" Updated by PlaneMVS's authors; """
# We removed `from apex import amp` and used amp from torch.cuda
# to support new version of PyTorch;
import torch
from torch.cuda import amp
# ------------------------------

class _ROIPool(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = input.size()
        output, argmax = _C.roi_pool_forward(
            input, roi, spatial_scale, output_size[0], output_size[1]
        )
        ctx.save_for_backward(input, roi, argmax)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, argmax = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_pool_backward(
            grad_output,
            input,
            rois,
            argmax,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
        )
        return grad_input, None, None, None


roi_pool = _ROIPool.apply


class ROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(ROIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    # ------------------------------
    #""" Updated by PlaneMVS's authors; """
    # ------------------------------
    #@amp.float_function # removed;
    @amp.autocast() # newly added by us;
    def forward(self, input, rois):
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ")"
        return tmpstr
