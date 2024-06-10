"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import os.path as osp

import argparse
import cv2

import numpy as np

import tqdm

import torch



""" load our own modules """
from src.config import cfg
from src.models.detector import build_detection_model
from src.utils.checkpoint import DetectronCheckpointer

from src.inference import (
    make_test_loader, vis_det, vis_depth, vis_depth_error, 
    plane_to_depth, plane_stereo_to_depth
    )
from src.inference.plane_stereo_to_depth import make_camera_grid
from src.inference.vis import transform_pointcloud


def inference(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)

    model.to(device).eval()

    checkpointer = DetectronCheckpointer(cfg, model)
    checkpointer.load(args.ckpt_file, use_latest=False, is_train=False)

    data_loader = make_test_loader(cfg)

    assert len(data_loader) == 1, 'Currently only support to infer on a single dataloader'
    data_loader = data_loader[0]

    with torch.no_grad():
        for sample_idx, sample in enumerate(tqdm.tqdm(data_loader)):
            if sample_idx == args.num_test:
                break

            images, targets, _ = sample
            result = model(images, targets)

            # merged_depth: planar depth for plane regions, single-network depth for non-planar regions
            merged_depth, instance_planar_depth, pixel_planar_depth, pred_instance_planes, gt_mask_planar_depth = plane_stereo_to_depth(result, images, targets)

            if merged_depth is not None:
                result[0].add_field('merged_depth', torch.from_numpy(merged_depth))

            if args.use_pixel_planar or merged_depth is None:
                result[0].add_field('planar_depth', torch.from_numpy(instance_planar_depth))

            else:
                result[0].add_field('planar_depth', torch.from_numpy(merged_depth))

            if result[0].has_field('refined_plane_map'):
                _, refined_instance_planar_depth, _, _, _ = plane_stereo_to_depth(result, images, targets, use_refine=True)

            else:
                refined_instance_planar_depth = None

            if args.visualize:
                label_mapping = targets.get_field('sem_mapping')
                net_label_mapping = dict()

                # the network starts from 1 for foreground and contains others
                for key, val in label_mapping.items():
                    net_label_mapping[key + 1] = val

                vis = vis_det(result, images['ori_ref_img'], net_label_mapping)
                vis = vis[:, 640:, ]

                if refined_instance_planar_depth is not None:
                    refined_depth_vis = vis_depth(refined_instance_planar_depth)
                    refined_depth_vis = np.asarray(refined_depth_vis)[..., ::-1]

                    vis = np.hstack([vis, refined_depth_vis])

                ori_ref_img = np.asarray(images['ori_ref_img'])[..., ::-1]

                if args.save_pnt_cloud:
                    img_intrinsic = targets.get_field('intrinsic')
                    transform_pointcloud(ori_ref_img, instance_planar_depth, img_intrinsic, 'pnt_' + args.save_dir, images['ref_path'])

                vis = np.hstack([ori_ref_img, vis])
                vis = cv2.resize(vis, None, fx=0.6, fy=0.6)

                scene_id, img_id = images['ref_path'].split('/')
                ref_img_id = img_id.split('.')[0]

                save_dir = osp.join(args.save_dir, scene_id)
                if not osp.exists(save_dir):
                    os.makedirs(save_dir)

                _, src_img_id = images['src_path'].split('/')
                cv2.imwrite(osp.join(save_dir, ref_img_id + '_' + src_img_id), vis)

    return


def main():
    parser = argparse.ArgumentParser(description="Pytorch PlaneRCNN Testing")

    parser.add_argument(
        "--config_file",
        required=True,
        help="path to config file",
        type=str
    )

    parser.add_argument(
        "--ckpt_file",
        required=True,
        help="the pretrained model directory loaded for testing"
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--visualize",
        help='whether to save and visualize the predictions',
        action='store_true'
    )

    parser.add_argument(
        "--save_dir",
        help="the dir to save the visualization images",
        default='plane_vis'
    )

    parser.add_argument(
        "--num_test",
        help="Sample to a upper bound number during testing or inference",
        type=int,
        default=100000
    )

    parser.add_argument(
        "--vis_depth_error",
        help="Visualize depth error grayscale map",
        action='store_true'
    )

    parser.add_argument(
        "--use_pixel_planar",
        help='Whether to use pixel planar map for depth evaluation',
        action='store_true'
    )

    parser.add_argument(
        "--vis_gt_mask",
        help='Whether to visualize the ground-truth plane masks as reference',
        action='store_true'
    )

    parser.add_argument(
        "--save_pnt_cloud",
        help='Whether to visualize and save the point cloud for plane reconstruction',
        action='store_true'
    )

    args = parser.parse_args()

    inference(args)

    return


if __name__ == '__main__':
    main()
