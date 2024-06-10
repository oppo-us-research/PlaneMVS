"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import os.path as osp
import tqdm

import argparse
import cv2

import numpy as np

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.inference import make_test_loader
from maskrcnn_benchmark.inference import vis_gt_det


def vis_gt(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    data_loader = make_test_loader(cfg)
    data_loader = data_loader[0]

    for sample_idx, sample in enumerate(tqdm.tqdm(data_loader)):
        images, targets, _ = sample

        gt_planar_semantic_map = targets.get_field('planar_semantic_map').cpu().numpy()
        gt_vis = vis_gt_det(targets, images['ori_ref_img'], gt_planar_semantic_map)

        save_folder = args.gt_save_dir
        if not osp.exists(save_folder):
            os.makedirs(save_folder)

        scene_id, img_id = images['ref_path'].split('/')
        _, src_img_id = images['src_path'].split('/')

        cv2.imwrite(osp.join(save_folder, scene_id + '_' + img_id + '_' + src_img_id), gt_vis)


def main():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        "--config_file",
        required=True,
        help='path to config file',
        type=str
    )

    parser.add_argument(
        "--gt_save_dir",
        help='the dir to save the visualization images',
        default='gt_mask_vis'
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    vis_gt(args)

    return


if __name__ == '__main__':
    main()
