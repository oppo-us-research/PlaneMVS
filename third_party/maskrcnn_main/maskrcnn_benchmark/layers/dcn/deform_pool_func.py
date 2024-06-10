# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# Copyright (c) 2024 OPPO. All rights reserved.
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------------
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# ------------------------------
""" Updated by PlaneMVS's authors; """
# Since we just build libs in the local `build`` dir,
# We do not install them in the system-level, like /usr/local/* 
# via something like `pip install maskrcnn_benchmark`;
# We explicitly load it from the local `build` dir in this project, 
# once the library was compiled. 
# You can run `./compile.sh` to compile the cuda codes 
# (including: maskrcnn_benchmark/csrc/cuda/*.cu);
#from maskrcnn_benchmark import _C # removed;
from build.lib.maskrcnn_benchmark import _C


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(
        ctx,
        data,
        rois,
        offset,
        spatial_scale,
        out_size,
        out_channels,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=.0
    ):
        ctx.spatial_scale = spatial_scale
        ctx.out_size = out_size
        ctx.out_channels = out_channels
        ctx.no_trans = no_trans
        ctx.group_size = group_size
        ctx.part_size = out_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std

        assert 0.0 <= ctx.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError

        n = rois.shape[0]
        output = data.new_empty(n, out_channels, out_size, out_size)
        output_count = data.new_empty(n, out_channels, out_size, out_size)
        _C.deform_psroi_pooling_forward(
            data,
            rois,
            offset,
            output,
            output_count,
            ctx.no_trans,
            ctx.spatial_scale,
            ctx.out_channels,
            ctx.group_size,
            ctx.out_size,
            ctx.part_size,
            ctx.sample_per_part,
            ctx.trans_std
        )

        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        ctx.output_count = output_count

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError

        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = None
        grad_offset = torch.zeros_like(offset)

        _C.deform_psroi_pooling_backward(
            grad_output,
            data,
            rois,
            offset,
            output_count,
            grad_input,
            grad_offset,
            ctx.no_trans,
            ctx.spatial_scale,
            ctx.out_channels,
            ctx.group_size,
            ctx.out_size,
            ctx.part_size,
            ctx.sample_per_part,
            ctx.trans_std
        )
        return (grad_input, grad_rois, grad_offset, None, None, None, None, None, None, None, None)


deform_roi_pooling = DeformRoIPoolingFunction.apply
