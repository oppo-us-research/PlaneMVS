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
from src.evaluation import evaluate_depth
from src.tools.export_dets import export_detections


def inference(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)

    model.to(device).eval()

    checkpointer = DetectronCheckpointer(cfg, model)
    checkpointer.load(args.ckpt_file, use_latest=False, is_train=False)

    if args.export_dets:
        data_loader = make_test_loader(cfg, is_train=True)

    else:
        data_loader = make_test_loader(cfg, is_train=False)

    assert len(data_loader) == 1, 'Currently only support to infer on a single dataloader'
    data_loader = data_loader[0]

    if args.eval:
        planar_depth_metrics = []

    with torch.no_grad():
        for sample_idx, sample in enumerate(tqdm.tqdm(data_loader)):
            if sample_idx == args.num_test:
                break

            images, targets, _ = sample
            result = model(images, targets)

            if args.export_dets:
                export_detections(result[0], images['ref_path'], save_dir='seven_scenes_pseudo_gt', dataset='7-scenes')
                continue

            if cfg.MODEL.METHOD == 'stereo':
                _, instance_planar_depth, pixel_planar_depth, pred_instance_planes, _ = plane_stereo_to_depth(result, images, targets)
                result[0].add_field('planar_depth', torch.from_numpy(instance_planar_depth))
                depth = instance_planar_depth

            else:
                depth, _, _ = plane_to_depth(result, targets)
                result[0].add_field('planar_depth', torch.from_numpy(depth))

            if args.eval:
                gt_depth = targets.get_field('depth')

                pred_planar_depth = result[0].get_field('planar_depth')
                single_planar_depth_metrics = evaluate_depth(gt_depth, pred_planar_depth)
                planar_depth_metrics.append(single_planar_depth_metrics)

                # pred_np_depth = result[0].get_field('depth')
                # single_np_depth_metrics = evaluate_depth(gt_depth, pred_np_depth)
                # np_depth_metrics.append(single_np_depth_metrics)

            if args.visualize:
                label_mapping = targets.get_field('sem_mapping')
                net_label_mapping = dict()

                # the network starts from 1 for foreground and contains others
                for key, val in label_mapping.items():
                    net_label_mapping[key + 1] = val

                vis = vis_det(result, images['ori_ref_img'], net_label_mapping)

                depth_vis = vis_depth(depth)
                # to adapt for opencv saving format
                depth_vis = np.asarray(depth_vis)[..., ::-1]

                vis = np.hstack([vis, depth_vis])

                if args.vis_with_gt:
                    depth_gt = targets.get_field('depth').cpu().numpy()
                    depth_gt_vis = vis_depth(depth_gt)
                    depth_gt_vis = np.asarray(depth_gt_vis)[..., ::-1]
                    vis = np.hstack([vis, depth_gt_vis])

                if args.vis_depth_error:
                    depth_error = np.abs(depth - depth_gt)
                    valid_mask = depth_gt > 1e-4
                    error_vis = vis_depth_error(depth_error, valid_mask)
                    vis = np.hstack([vis, error_vis])

                ori_ref_img = np.asarray(images['ori_ref_img'])[..., ::-1]
                ori_src_img = np.asarray(images['ori_src_img'])[..., ::-1]
                vis = np.hstack([vis, ori_ref_img, ori_src_img])

                vis = cv2.resize(vis, None, fx=0.7, fy=0.7)

                if not osp.exists(args.save_dir):
                    os.makedirs(args.save_dir)

                scene_id, seq_id, img_id = images['ref_path'].split('/')

                cv2.imwrite(osp.join(args.save_dir, scene_id + '_' + seq_id  + '_' + img_id), vis)

    if args.eval:
        planar_depth_metrics = np.asarray(planar_depth_metrics)
        mean_planar_depth_metrics = np.mean(planar_depth_metrics, axis=0)
        mean_planar_depth_metrics = np.round(mean_planar_depth_metrics, 3)
        print('Mean Planar Depth Metrics:', mean_planar_depth_metrics)

        # np_depth_metrics = np.asarray(np_depth_metrics)
        # mean_np_depth_metrics = np.mean(np_depth_metrics, axis=0)
        # mean_np_depth_metrics = np.round(mean_np_depth_metrics, 3)
        # print('Mean Non-Planar Depth Metrics:', mean_np_depth_metrics)

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
        "--eval",
        action="store_true",
        help="whether to evaluate the testing images during inference"
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
        "--vis_with_gt",
        help="Visualize gt as comparison",
        action="store_true"
    )

    parser.add_argument(
        "--vis_depth_error",
        help='Visualize depth error grayscale map',
        action="store_true"
    )

    parser.add_argument(
        "--export_dets",
        help="Export Detection results as .npy format for other use",
        action="store_true"
    )

    args = parser.parse_args()

    inference(args)

    return


if __name__ == '__main__':
    main()
