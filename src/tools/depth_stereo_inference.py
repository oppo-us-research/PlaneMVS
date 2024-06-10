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
from src.inference import (make_test_loader, vis_det, 
                           vis_gt_det, vis_depth, 
                           vis_depth_error, vis_disparity,
                           get_planar_semantic_map,
                           )
from src.evaluation import (evaluate_detection, evaluate_depth, 
                            evaluate_semantic, evaluate_masks,
                            evaluate_map, accumulate_map,
                            )


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

    num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES - 1

    wo_plane_det = cfg.MODEL.STEREO.WO_DET_LOSS
    eval_depth_only = args.eval_depth_only or wo_plane_det

    if args.eval:
        APs = []
        init_depth_metrics = []
        refined_depth_metrics = []

        planar_depth_metrics = []
        gt_plane_area_depth_metrics = []

        seg_metrics = []

        mAPs = [[] for _ in range(num_classes)]

        pred_img_semantics = []
        gt_img_semantics = []

    with torch.no_grad():
        for sample_idx, sample in enumerate(tqdm.tqdm(data_loader)):
            if sample_idx == args.num_test:
                break

            images, targets, _ = sample
            result = model(images, targets)

            if args.eval:
                if not eval_depth_only:
                    single_APs, _, _, pos_pred_idxs, matched_gt_idxs = evaluate_detection(result, targets)
                    if single_APs is not None:
                        APs.append(single_APs)

                    for label in range(num_classes):
                        cls_res = evaluate_map(result, targets, label)

                        if cls_res is not None:
                            mAPs[label].append(cls_res)

                    single_seg_metrics = evaluate_masks(result, targets)
                    seg_metrics.append(single_seg_metrics)

                gt_depth = targets.get_field('depth')
                gt_masks = targets.get_field('masks').instances.masks
                gt_plane_mask = np.sum(gt_masks.numpy(), axis=0) > 0

                if result[0].has_field('depth'):
                    init_depth = result[0].get_field('depth')
                    init_depth = init_depth.cpu().numpy()

                    single_init_depth_metrics = evaluate_depth(gt_depth, init_depth)
                    init_depth_metrics.append(single_init_depth_metrics)

                if result[0].has_field('refined_depth'):
                    refined_depth = result[0].get_field('refined_depth')
                    refined_depth = refined_depth.cpu().numpy()

                    single_refined_depth_metrics = evaluate_depth(gt_depth, refined_depth)
                    refined_depth_metrics.append(single_refined_depth_metrics)

                if result[0].has_field('planar_depth'):
                    planar_depth = result[0].get_field('planar_depth')
                    planar_depth = planar_depth.cpu().numpy()

                    single_planar_depth_metrics = evaluate_depth(gt_depth, planar_depth)
                    planar_depth_metrics.append(single_planar_depth_metrics)

                    single_gt_plane_area_depth_metrics = evaluate_depth(gt_depth, planar_depth, gt_plane_mask)
                    gt_plane_area_depth_metrics.append(single_gt_plane_area_depth_metrics)

                if not eval_depth_only:
                    pred_labels = result[0].get_field('labels').cpu().numpy()
                    pred_masks = result[0].get_field('mask').cpu().numpy()
                    planar_semantic_map = get_planar_semantic_map(pred_labels, pred_masks)
                    gt_planar_semantic_map = targets.get_field('planar_semantic_map').cpu().numpy()

                    pred_img_semantics.append(planar_semantic_map)
                    gt_img_semantics.append(gt_planar_semantic_map)

            if args.visualize:
                if args.vis_disparity:
                    depth_vis_func = vis_disparity

                else:
                    depth_vis_func = vis_depth

                label_mapping = targets.get_field('sem_mapping')
                net_label_mapping = dict()

                # the network starts from 1 for foreground and contains others
                for key, val in label_mapping.items():
                    net_label_mapping[key + 1] = val

                vis = vis_det(result, images['ori_ref_img'], net_label_mapping)

                if not eval_depth_only and args.vis_gt_mask:
                    gt_vis = vis_gt_det(targets, images['ori_ref_img'], gt_planar_semantic_map)
                    vis = np.hstack([vis, gt_vis])

                max_depth = np.max(gt_depth.cpu().numpy())

                depth_vis = depth_vis_func(init_depth, max_depth=max_depth)
                # to adapt for opencv saving format
                depth_vis = np.asarray(depth_vis)[..., ::-1]

                vis = np.hstack([vis, depth_vis])

                if refined_depth is not None:
                    refined_depth_vis = depth_vis_func(refined_depth, max_depth=max_depth)
                    refined_depth_vis = np.asarray(refined_depth_vis)[..., ::-1]

                    vis = np.hstack([vis, refined_depth_vis])

                if args.vis_with_gt:
                    depth_gt = targets.get_field('depth').cpu().numpy()
                    depth_gt_vis = depth_vis_func(depth_gt, max_depth=max_depth)
                    depth_gt_vis = np.asarray(depth_gt_vis)[..., ::-1]
                    vis = np.hstack([vis, depth_gt_vis])

                if args.vis_depth_error:
                    depth_error = np.abs(init_depth - depth_gt)
                    valid_mask = depth_gt > 1e-4
                    error_vis = vis_depth_error(depth_error, valid_mask)
                    vis = np.hstack([vis, error_vis])

                    if refined_depth is not None:
                        refined_depth_error = np.abs(refined_depth - depth_gt)
                        valid_mask = depth_gt > 1e-4
                        refined_error_vis = vis_depth_error(refined_depth_error, valid_mask)
                        vis = np.hstack([vis, refined_error_vis])

                ori_ref_img = np.asarray(images['ori_ref_img'])[..., ::-1]
                ori_src_img = np.asarray(images['ori_src_img'])[..., ::-1]

                vis = np.hstack([vis, ori_ref_img, ori_src_img])

                vis = cv2.resize(vis, None, fx=0.6, fy=0.6)

                if not osp.exists(args.save_dir):
                    os.makedirs(args.save_dir)

                scene_id, img_id = images['ref_path'].split('/')
                ref_img_id = img_id.split('.')[0]

                _, src_img_id = images['src_path'].split('/')
                cv2.imwrite(osp.join(args.save_dir, scene_id + '_' + ref_img_id + '_' + src_img_id), vis)

    if args.eval:
        if not eval_depth_only:
            APs = np.asarray(APs)
            mean_AP = np.mean(APs, axis=0)
            mean_AP = np.round(mean_AP, 3)
            print('Mean AP:', mean_AP)

            mean_mAP = accumulate_map(num_classes, mAPs)
            print('Mean mAP:', mean_mAP)

            seg_metrics = np.asarray(seg_metrics)
            mean_seg_metrics = np.mean(seg_metrics, 0)
            mean_seg_metrics = np.round(mean_seg_metrics, 3)
            print('Mean PlaneSeg:', mean_seg_metrics)

        if len(init_depth_metrics) > 0:
            init_depth_metrics = np.asarray(init_depth_metrics)
            mean_init_depth_metrics = np.mean(init_depth_metrics, axis=0)
            mean_init_depth_metrics = np.round(mean_init_depth_metrics, 3)
            print('Mean Initial Depth Metrics for Whole Img:', mean_init_depth_metrics)

        if len(refined_depth_metrics) > 0:
            refined_depth_metrics = np.asarray(refined_depth_metrics)
            mean_refined_depth_metrics = np.mean(refined_depth_metrics, axis=0)
            mean_refined_depth_metrics = np.round(mean_refined_depth_metrics, 3)
            print('Mean Refined Depth Metrics for Whole Img:', mean_refined_depth_metrics)

        if len(planar_depth_metrics) > 0:
            planar_depth_metrics = np.asarray(planar_depth_metrics)
            mean_planar_depth_metrics = np.mean(planar_depth_metrics, axis=0)
            mean_planar_depth_metrics = np.round(mean_planar_depth_metrics, 3)
            print('Mean Planar Depth Metrics for Whole Img:', mean_planar_depth_metrics)

        if len(gt_plane_area_depth_metrics) > 0:
            gt_plane_area_depth_metrics = np.asarray(gt_plane_area_depth_metrics)
            mean_gt_plane_area_depth_metrics = np.mean(gt_plane_area_depth_metrics, axis=0)
            mean_gt_plane_area_depth_metrics = np.round(mean_gt_plane_area_depth_metrics, 3)
            print('Mean Gt Plane Area Depth Metrics:', mean_gt_plane_area_depth_metrics)

        if not eval_depth_only:
            label_mapping = targets.get_field('sem_mapping')
            n_class = len(label_mapping.keys()) + 1

            img_semantic_metrics = evaluate_semantic(gt_img_semantics, pred_img_semantics, n_class)
            print('=' * 10, 'img semantics(consider every pixels)', '=' * 10)
            print(img_semantic_metrics)

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
        "--vis_np_depth",
        help="if set, visualize the non-planar depth output by the depth network",
        action='store_true'
    )

    parser.add_argument(
        "--vis_with_gt",
        help="Visualize gt as comparison",
        action='store_true'
    )

    parser.add_argument(
        "--vis_depth_error",
        help="Visualize depth error grayscale map",
        action='store_true'
    )

    parser.add_argument(
        "--vis_disparity",
        help="Visualize disparity in place of depth map",
        action='store_true'
    )

    parser.add_argument(
        "--vis_pixel_planar",
        help="Visualize pixel planar depth map",
        action='store_true'
    )

    parser.add_argument(
        "--vis_uncertainty",
        help="Visualize the pred uncertainty",
        action="store_true"
    )

    parser.add_argument(
        "--vis_mask_uncertainty",
        help="Visualize the spatial uncertainty for mask loss",
        action="store_true"
    )

    parser.add_argument(
        "--use_pixel_planar",
        help='Whether to use pixel planar map for depth evaluation',
        action='store_true'
    )

    parser.add_argument(
        "--vis_src_result",
        help='Whether to visualize the detections in source image',
        action="store_true"
    )

    parser.add_argument(
        "--merge_det_result",
        help="Whether to merge the det results from two images by their correspondence",
        action="store_true"
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

    parser.add_argument(
        '--eval_depth_only',
        help='Whether to only evaluate depth metrics and skip the plane detection metrics',
        action='store_true'
    )

    args = parser.parse_args()

    inference(args)

    return


if __name__ == '__main__':
    main()
