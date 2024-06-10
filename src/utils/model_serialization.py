"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
import logging

import torch

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_serialization import strip_prefix_if_present


# ------------------------------
# Updated by PlaneMVS's authors;
# ------------------------------
def align_and_update_state_dicts(model_state_dict, loaded_state_dict, 
                                 # newly added args;
                                 log_model=False, 
                                 ignore_keys=[]
                                 ):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]

        if log_model:
            logger.info(
                log_str_template.format(
                    key,
                    max_size,
                    key_old,
                    max_size_loaded,
                    tuple(loaded_state_dict[key_old].shape),
                )
            )

    skipped_keys = []

    for ignore_key in ignore_keys:
        for key in model_state_dict.keys():
            if ignore_key in key:
                skipped_keys.append(key)

    for key in skipped_keys:
        model_state_dict.pop(key)


# ------------------------------
# Updated by PlaneMVS's authors;
# ------------------------------
def load_state_dict(cfg, model, loaded_state_dict, 
                    # newly added args
                    tune_from_scratch=False, 
                    resume=False, 
                    is_train=True, 
                    ignore_keys=['predictor']
                    ):
    model_state_dict = model.state_dict()

    # if tune from scratch or resume, the model parameter shape should be aligned
    if resume or not is_train:
        ignore_keys = []

    elif tune_from_scratch and not cfg.MODEL.METHOD == 'stereo':
        ignore_keys = []

    elif tune_from_scratch and is_train and cfg.MODEL.METHOD == 'stereo':
        ignore_keys = ['loss_term_uncert']

    elif cfg.MODEL.METHOD == 'srcnn':
        ignore_keys = ['rpn.head.cls_logits', 'rpn.head.bbox_pred', 'roi_heads.box.feature_extractor.fc6', 'predictor']

        if cfg.MODEL.SRCNN.SEPARATE_PRED:
            ignore_keys.append('roi_heads.mask.feature_extractor.mask_fcn1')

    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, ignore_keys=ignore_keys)

    if (tune_from_scratch and not cfg.MODEL.METHOD == 'stereo') or resume or not is_train:
        strict = True

    else:
        strict = False

    # use strict loading
    model.load_state_dict(model_state_dict, strict=strict)
