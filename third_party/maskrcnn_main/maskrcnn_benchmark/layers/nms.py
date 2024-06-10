# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# Copyright (c) 2024 OPPO. All rights reserved.
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

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
nms = _C.nms
# ------------------------------
