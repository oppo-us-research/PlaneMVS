"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

from inspect import currentframe, getframeinfo
from termcolor import colored
import numpy as np
import torch
import torchvision.utils as vutils


# In order to correctly import modules under
# `third_party/maskrcnn_main/maskrcnn_benchmark`,
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
from maskrcnn_benchmark.structures.image_list import ImageList

def print_indebugmode(message):
    previous_frame = currentframe().f_back
    (
        filename,
        line_number,
        function_name,
        lines,
        index,
    ) = getframeinfo(previous_frame)
    print (colored("[*** 4Debug] ", 'red') + 
        filename + ':' + str(line_number) + ' - ', message)


def check_nan_inf(inp, name):
    assert not torch.isnan(inp).any(), \
        "Found Nan in input {}, shape = {}, val = {}".format(
            name, inp.shape, inp)
    assert not torch.isinf(inp).any(), \
        "Found Inf in input {}, shape = {}, val = {}".format(
            name, inp.shape, inp)


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)
    return wrapper


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper

@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))

@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    # ImageList;
    elif isinstance(vars, ImageList):
        return ImageList(vars.tensors.cuda(), vars.image_sizes)
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))

@make_recursive_func
def check_allfloat(vars):
    assert isinstance(vars, float)


def save_scalars(logger, mode_tag, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for tag, values in scalar_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            scalar_name = '{}/{}'.format(mode_tag, tag)
            # if len(values) > 1:
            scalar_name = scalar_name + "_" + str(idx)
            #print ("save scalar ", scalar_name)
            logger.add_scalar(scalar_name, value, global_step)


def save_images(logger, mode_tag, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)
    for tag, values in images_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            if len(value.shape) == 3:
                value = value[:, np.newaxis, :, :]
            value = value[:1]
            value = torch.from_numpy(value)

            image_name = '{}/{}'.format(mode_tag, tag)
            if len(values) > 1:
                image_name = image_name + "_" + str(idx)
            logger.add_image(image_name, 
                    vutils.make_grid(value, padding=0, nrow=1, normalize=True, scale_each=True),
                    global_step
                )