"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch


""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.imports import import_file

""" load our own modules """
from src.datasets.build import build_dataset


def make_test_loader(cfg, torchloader=False, is_train=False):
    transforms = build_transforms(cfg, is_train=is_train)

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )

    DatasetCatalog = paths_catalog.DatasetCatalog

    if is_train:
        dataset_list = cfg.DATASETS.TRAIN

    else:
        dataset_list = cfg.DATASETS.TEST

    datasets = build_dataset(dataset_list, transforms, DatasetCatalog, is_train=is_train)

    data_loaders = []

    if torchloader:
        for dataset in datasets:
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                drop_last=False
            )

            data_loaders.append(data_loader)

    else:
        data_loaders = datasets

    return data_loaders
