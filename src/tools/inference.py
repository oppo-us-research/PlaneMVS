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
from src.inference.plane_stereo_to_depth import make_camera_grid
from src.inference.vis import transform_pointcloud
from src.inference import (make_test_loader, vis_det, 
                           vis_gt_det, vis_depth, 
                           vis_depth_error, vis_disparity,
                           plane_to_depth,
                           get_planar_semantic_map,
                           )
from src.evaluation import (evaluate_detection, evaluate_depth, 
                            evaluate_semantic, evaluate_masks,
                            evaluate_map, accumulate_map,
                            evaluate_plane_params,
                            evaluate_plane_geometric
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

    if args.eval:
        APs = []
        refined_APs = []

        seg_metrics = []
        refined_seg_metrics = []

        mAPs = [[] for _ in range(num_classes)]

        planar_depth_metrics = []
        np_depth_metrics = []
        gt_planar_area_depth_metrics = []

        pred_planar_semantics = []
        gt_planar_semantics = []

        pred_img_semantics = []
        gt_img_semantics = []

        plane_normal_diffs = []
        plane_offset_diffs = []
        plane_n_div_d_diffs = []

        plane_param_diffs = []
        weighted_diffs = []
        gt_areas = []

    with torch.no_grad():
        for sample_idx, sample in enumerate(tqdm.tqdm(data_loader)):
            if sample_idx == args.num_test:
                break

            images, targets, _ = sample
            result = model(images, targets)

            depth, pred_plane_mask, pred_instance_planes = plane_to_depth(result, targets)
            result[0].add_field('planar_depth', torch.from_numpy(depth))

            if args.eval:
                single_APs, _, _, pos_pred_idxs, matched_gt_idxs = evaluate_detection(result, targets)
                if single_APs is not None:
                    APs.append(single_APs)

                for label in range(num_classes):
                    cls_res = evaluate_map(result, targets, label)
                    mAPs[label].append(cls_res)

                if not cfg.MODEL.METHOD == 'consistency' and not cfg.MODEL.ROI_BOX_HEAD.WO_NORMAL_HEAD:
                    normal_diff, offset_diff, n_div_d_diff = evaluate_plane_params(pred_instance_planes, targets, pos_pred_idxs, matched_gt_idxs)
                    if normal_diff is not None:
                        plane_normal_diffs.append(normal_diff)
                        plane_offset_diffs.append(offset_diff)
                        plane_n_div_d_diffs.append(n_div_d_diff)

                single_seg_metrics = evaluate_masks(result, targets)
                seg_metrics.append(single_seg_metrics)

                if result[0].has_field('refined_mask'):
                    single_refined_APs, _, _, _, _ = evaluate_detection(result, targets, use_refine=True)
                    refined_APs.append(single_refined_APs)

                    single_refined_seg_metrics = evaluate_masks(result, targets, use_refine=True)
                    refined_seg_metrics.append(single_refined_seg_metrics)

                gt_depth = targets.get_field('depth')

                h, w = gt_depth.size()[-2:]

                pred_planar_depth = result[0].get_field('planar_depth')
                single_planar_depth_metrics = evaluate_depth(gt_depth, pred_planar_depth)
                planar_depth_metrics.append(single_planar_depth_metrics)

                intrinsic = targets.get_field('intrinsic')[:3, :3]
                _, camera_grid, _ = make_camera_grid(intrinsic, h, w)
                pred_planar_depth = pred_planar_depth.cpu().numpy()

                single_param_diff, single_weighted_diff, single_area = evaluate_plane_geometric(pred_planar_depth, camera_grid, targets)
                if single_param_diff is not None:
                    plane_param_diffs.append(single_param_diff)
                    weighted_diffs.append(single_weighted_diff)
                    gt_areas.append(single_area)

                pred_np_depth = result[0].get_field('depth')
                single_np_depth_metrics = evaluate_depth(gt_depth, pred_np_depth)
                np_depth_metrics.append(single_np_depth_metrics)

                gt_plane_mask = targets.get_field('masks').instances.masks.sum(0) > 0
                gt_plane_mask = gt_plane_mask.cpu().numpy()

                single_gt_planar_area_depth_metrics = evaluate_depth(gt_depth, pred_planar_depth, valid_mask=gt_plane_mask)
                if single_gt_planar_area_depth_metrics is not None:
                    gt_planar_area_depth_metrics.append(single_gt_planar_area_depth_metrics)

                pred_labels = result[0].get_field('labels').cpu().numpy()
                pred_masks = result[0].get_field('mask').cpu().numpy()

                planar_semantic_map = get_planar_semantic_map(pred_labels, pred_masks)
                gt_planar_semantic_map = targets.get_field('planar_semantic_map').cpu().numpy()

                pred_img_semantics.append(planar_semantic_map)
                gt_img_semantics.append(gt_planar_semantic_map)

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

                vis = cv2.resize(vis, None, fx=0.6, fy=0.6)

                if not osp.exists(args.save_dir):
                    os.makedirs(args.save_dir)

                scene_id, img_id = images['ref_path'].split('/')

                cv2.imwrite(osp.join(args.save_dir, scene_id + '_' + img_id), vis)

                if args.save_pnt_cloud:
                    img_intrinsic = targets.get_field('intrinsic')
                    ori_ref_img = np.asarray(images['ori_ref_img'])[..., ::-1]
                    points = transform_pointcloud(ori_ref_img, depth, img_intrinsic, 'pnt_' + args.save_dir, images['ref_path'])

                    gt_depth = gt_depth.cpu().numpy()
                    gt_points = transform_pointcloud(ori_ref_img, gt_depth, img_intrinsic, 'gt_pnt_' + args.save_dir, images['ref_path'])

    if args.eval:
        APs = np.asarray(APs)
        mean_AP = np.mean(APs, axis=0)
        mean_AP = np.round(mean_AP, 3)
        print('Mean AP:', mean_AP)

        mAPs = accumulate_map(num_classes, mAPs)

        seg_metrics = np.asarray(seg_metrics)
        mean_seg_metrics = np.mean(seg_metrics, axis=0)
        mean_seg_metrics = np.round(mean_seg_metrics, 3)
        print('Mean PlaneSeg:', mean_seg_metrics)

        if len(refined_APs) > 0:
            refined_APs = np.asarray(refined_APs)
            mean_refined_AP = np.mean(refined_APs, axis=0)
            mean_refined_AP = np.round(mean_refined_AP, 3)
            print('Mean Refined AP:', mean_refined_AP)

            refined_seg_metrics = np.asarray(refined_seg_metrics)
            mean_refined_seg_metrics = np.mean(refined_seg_metrics, axis=0)
            mean_refined_seg_metrics = np.round(mean_refined_seg_metrics, 3)
            print('Mean Refined PlaneSeg:', mean_refined_seg_metrics)

        planar_depth_metrics = np.asarray(planar_depth_metrics)
        mean_planar_depth_metrics = np.mean(planar_depth_metrics, axis=0)
        mean_planar_depth_metrics = np.round(mean_planar_depth_metrics, 3)
        print('Mean Planar Depth Metrics:', mean_planar_depth_metrics)

        np_depth_metrics = np.asarray(np_depth_metrics)
        mean_np_depth_metrics = np.mean(np_depth_metrics, axis=0)
        mean_np_depth_metrics = np.round(mean_np_depth_metrics, 3)
        print('Mean Non-Planar Depth Metrics:', mean_np_depth_metrics)

        gt_planar_area_depth_metrics = np.asarray(gt_planar_area_depth_metrics)
        mean_gt_planar_area_depth_metrics = np.mean(gt_planar_area_depth_metrics, axis=0)
        mean_gt_planar_area_depth_metrics = np.round(mean_gt_planar_area_depth_metrics, 3)
        print('Mean Gt-Planar-Area Depth Metrics:', mean_gt_planar_area_depth_metrics)

        label_mapping = targets.get_field('sem_mapping')
        n_class = len(label_mapping.keys()) + 1

        # planar_semantic_metrics = evaluate_semantic(gt_planar_semantics, pred_planar_semantics, n_class)
        # print('=' * 10, 'planar semantics(only consider those detected pixels):', '=' * 10)
        # print(planar_semantic_metrics)

        img_semantic_metrics = evaluate_semantic(gt_img_semantics, pred_img_semantics, n_class)
        print('=' * 10, 'img semantics(consider every pixels)', '=' * 10)
        print(img_semantic_metrics)

        mean_normal_diff = np.round(np.mean(plane_normal_diffs, axis=0), 3)
        mean_offset_diff = np.round(np.mean(plane_offset_diffs, axis=0), 3)
        mean_n_div_d_diff = np.round(np.mean(plane_n_div_d_diffs, axis=0), 3)

        print('=' * 10, 'Mean Plane Normal Diff:', '=' * 10)
        print(mean_normal_diff)
        print('=' * 10, 'Mean Plane Offset Diff:', '=' * 10)
        print(mean_offset_diff)
        print('=' * 10, 'Mean Plane n/d Diff:', '=' * 10)
        print(mean_n_div_d_diff)

        mean_plane_param_diff = np.mean(np.concatenate(plane_param_diffs, axis=0))
        weighted_mean_plane_param_diff = np.sum(weighted_diffs) / np.sum(gt_areas)
        print('=' * 10, 'Mean Plane Geometric Diff:', '=' * 10)
        print(round(mean_plane_param_diff, 3))

        median_plane_param_diff = np.median(np.concatenate(plane_param_diffs, axis=0))
        print('=' * 10, 'Median Plane Geometric Diff:', '=' * 10)
        print(round(median_plane_param_diff, 3))

        print('=' * 10, 'Weighted Mean Plane Geometric Diff:', '=' * 10)
        print(round(weighted_mean_plane_param_diff, 3))

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
        "--save_pnt_cloud",
        help="Whether to visualize and save the point cloud for plane reconstruction",
        action='store_true'
    )

    args = parser.parse_args()

    inference(args)

    return


if __name__ == '__main__':
    main()
