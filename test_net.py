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

import sys

### In order to correctly call third_party/maskrcnn_main/maskrcnn_benchmark ###
### modules, and do not destory the original import format ###
### inside the maskrcnn_benchmark python files ###
sys.path.append('third_party/maskrcnn_main')

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
from maskrcnn_benchmark.structures.bounding_box import BoxList


""" load our own moduels """
from src.config import cfg
from src.models.detector import build_detection_model
from src.utils.checkpoint import DetectronCheckpointer
from src.inference import make_test_loader
from src.inference import vis_det, vis_gt_det, vis_depth, vis_depth_error
from src.inference import vis_disparity

from src.inference import plane_to_depth, plane_stereo_to_depth
from src.inference import get_planar_semantic_map
from src.inference.vis import transform_pointcloud

from src.evaluation import evaluate_detection
from src.evaluation import evaluate_depth
from src.evaluation import evaluate_semantic
from src.evaluation import evaluate_masks
from src.evaluation import evaluate_plane_geometric
from src.evaluation import evaluate_plane_params
from src.evaluation import evaluate_map, accumulate_map


def inference(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    print (f"[***] inference device = {device}")

    model.to(device).eval()

    checkpointer = DetectronCheckpointer(cfg, model)

    # will load the ckpt from `args.ckpt_file`
    checkpointer.load(args.ckpt_file, use_latest=False, is_train=False)

    data_loader = make_test_loader(cfg, torchloader=False, is_train=False)
    #data_loader = make_test_loader(cfg, torchloader=True, is_train=False)

    assert len(data_loader) == 1, 'Currently only support to infer on a single dataloader'
    data_loader = data_loader[0]

    num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES - 1

    if args.eval:
        APs = []
        planar_depth_metrics = []
        np_depth_metrics = []

        seg_metrics = []

        mAPs = [[] for _ in range(num_classes)]

        img_depth_metrics = []
        gt_mask_img_depth_metrics = []
        img_pixel_planar_depth_metrics = []

        planar_area_depth_metrics = []
        planar_area_np_depth_metrics = []

        gt_planar_area_depth_metrics = []
        gt_planar_area_np_depth_metrics = []

        refined_gt_planar_area_depth_metrics = []

        gt_planar_area_pixel_depth_metrics = []

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
            if targets is not None: 
                if isinstance(targets, (list, tuple)):
                    targets = [target.to(device) for target in targets]
                elif isinstance(targets, BoxList):
                    targets = targets.to(device)
                elif isinstance(targets, torch.Tensor):
                    targets = targets.to(device)
                else:
                    raise NotImplementedError
            result = model(images, targets)

            # merged_depth: planar depth for plane regions, single-network depth for non-planar regions
            (merged_depth, 
            instance_planar_depth, 
            pixel_planar_depth, 
            pred_instance_planes, 
            gt_mask_planar_depth) = plane_stereo_to_depth(result, images, targets)

            if merged_depth is not None:
                result[0].add_field('merged_depth', torch.from_numpy(merged_depth))

            if args.use_pixel_planar or merged_depth is None:
                result[0].add_field('planar_depth', torch.from_numpy(instance_planar_depth))

            else:
                result[0].add_field('planar_depth', torch.from_numpy(merged_depth))

            if result[0].has_field('refined_plane_map'):
                _, refined_instance_planar_depth, _, refined_pred_instance_planes, _ = plane_stereo_to_depth(result, images, targets, use_refine=True)

            else:
                refined_instance_planar_depth = None

            if args.eval:
                if not args.eval_depth_only:
                    single_APs, _, _, pos_pred_idxs, matched_gt_idxs = evaluate_detection(result, targets)
                    if single_APs is not None:
                        APs.append(single_APs)

                    for label in range(num_classes):
                        #print ("???? evaluate_map targets = ", targets)
                        cls_res = evaluate_map(result, targets, label)

                        if cls_res is not None:
                            mAPs[label].append(cls_res)

                    single_seg_metrics = evaluate_masks(result, targets)
                    seg_metrics.append(single_seg_metrics)

                    normal_diff, offset_diff, n_div_d_diff = evaluate_plane_params(pred_instance_planes, targets, pos_pred_idxs, matched_gt_idxs)
                    if normal_diff is not None:
                        plane_normal_diffs.append(normal_diff)
                        plane_offset_diffs.append(offset_diff)
                        plane_n_div_d_diffs.append(n_div_d_diff)

                    camera_grid = result[0].get_field('camera_grid')
                    if camera_grid is not None:
                        single_param_diff, single_weighted_diff, single_area = evaluate_plane_geometric(instance_planar_depth, camera_grid, targets)
                        if single_param_diff is not None:
                            plane_param_diffs.append(single_param_diff)
                            weighted_diffs.append(single_weighted_diff)
                            gt_areas.append(single_area)

                gt_depth = targets.get_field('depth')

                if result[0].has_field('merged_depth'):
                    pred_planar_depth = result[0].get_field('merged_depth')
                    single_planar_depth_metrics = evaluate_depth(gt_depth, pred_planar_depth)
                    planar_depth_metrics.append(single_planar_depth_metrics)

                single_img_depth_metrics = evaluate_depth(gt_depth, instance_planar_depth)
                img_depth_metrics.append(single_img_depth_metrics)

                single_img_pixel_planar_depth_metrics = evaluate_depth(gt_depth, pixel_planar_depth)
                img_pixel_planar_depth_metrics.append(single_img_pixel_planar_depth_metrics)

                if not args.eval_depth_only and gt_mask_planar_depth is not None:
                    single_img_gt_mask_depth_metrics = evaluate_depth(gt_depth, gt_mask_planar_depth)
                    gt_mask_img_depth_metrics.append(single_img_gt_mask_depth_metrics)

                if result[0].has_field('depth'):
                    pred_np_depth = result[0].get_field('depth')
                    single_np_depth_metrics = evaluate_depth(gt_depth, pred_np_depth)
                    np_depth_metrics.append(single_np_depth_metrics)

                if not args.use_pixel_planar and merged_depth is not None:
                    instance_planar_depth = merged_depth

                pred_plane_mask = result[0].get_field('pred_plane_mask')

                single_planar_area_depth_metrics = evaluate_depth(gt_depth, instance_planar_depth, valid_mask=pred_plane_mask)
                if single_planar_area_depth_metrics is not None:
                    planar_area_depth_metrics.append(single_planar_area_depth_metrics)

                if result[0].has_field('depth'):
                    single_planar_area_np_depth_metrics = evaluate_depth(gt_depth, pred_np_depth, valid_mask=pred_plane_mask)
                    if single_planar_area_np_depth_metrics is not None:
                        planar_area_np_depth_metrics.append(single_planar_area_np_depth_metrics)

                if not args.eval_depth_only:
                    gt_plane_mask = targets.get_field('gt_plane_mask')

                    single_gt_planar_area_depth_metrics = evaluate_depth(gt_depth, instance_planar_depth, valid_mask=gt_plane_mask)
                    if single_gt_planar_area_depth_metrics is not None:
                        gt_planar_area_depth_metrics.append(single_gt_planar_area_depth_metrics)

                    if result[0].has_field('depth'):
                        single_gt_planar_area_np_depth_metrics = evaluate_depth(gt_depth, pred_np_depth, valid_mask=gt_plane_mask)
                        if single_gt_planar_area_np_depth_metrics is not None:
                            gt_planar_area_np_depth_metrics.append(single_gt_planar_area_np_depth_metrics)

                    if refined_instance_planar_depth is not None:
                        single_refined_gt_planar_area_depth_metrics = evaluate_depth(gt_depth, refined_instance_planar_depth, valid_mask=gt_plane_mask)
                        refined_gt_planar_area_depth_metrics.append(single_refined_gt_planar_area_depth_metrics)

                    single_gt_planar_area_pixel_depth_metrics = evaluate_depth(gt_depth, pixel_planar_depth, valid_mask=gt_plane_mask)
                    if single_gt_planar_area_pixel_depth_metrics is not None:
                        gt_planar_area_pixel_depth_metrics.append(single_gt_planar_area_pixel_depth_metrics)

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

                if not args.eval_depth_only and args.vis_gt_mask:
                    gt_vis = vis_gt_det(targets, images['ori_ref_img'], gt_planar_semantic_map)
                    vis = np.hstack([vis, gt_vis])

                max_depth = np.max(gt_depth.cpu().numpy())

                depth_vis = depth_vis_func(instance_planar_depth, max_depth=max_depth)
                # to adapt for opencv saving format
                depth_vis = np.asarray(depth_vis)[..., ::-1]

                vis = np.hstack([vis, depth_vis])

                if refined_instance_planar_depth is not None and args.vis_refine:
                    refined_depth_vis = depth_vis_func(refined_instance_planar_depth, max_depth=max_depth)
                    refined_depth_vis = np.asarray(refined_depth_vis)[..., ::-1]

                    vis = np.hstack([vis, refined_depth_vis])

                if args.vis_np_depth:
                    pred_np_depth = pred_np_depth.squeeze().cpu().numpy()
                    pred_np_depth_vis = depth_vis_func(pred_np_depth, max_depth=max_depth)
                    pred_np_depth_vis = np.asarray(pred_np_depth_vis)[..., ::-1]
                    vis = np.hstack([vis, pred_np_depth_vis])

                if args.vis_pixel_planar:
                    pixel_planar_depth_vis = depth_vis_func(pixel_planar_depth, max_depth=max_depth)
                    pixel_planar_depth_vis = np.asarray(pixel_planar_depth_vis)[..., ::-1]
                    vis = np.hstack([vis, pixel_planar_depth_vis])

                if args.vis_with_gt:
                    depth_gt = targets.get_field('depth').cpu().numpy()
                    depth_gt_vis = depth_vis_func(depth_gt, max_depth=max_depth)
                    depth_gt_vis = np.asarray(depth_gt_vis)[..., ::-1]
                    vis = np.hstack([vis, depth_gt_vis])

                if args.vis_depth_error:
                    depth_error = np.abs(instance_planar_depth - depth_gt)
                    valid_mask = depth_gt > 1e-4
                    error_vis = vis_depth_error(depth_error, valid_mask)
                    vis = np.hstack([vis, error_vis])

                    if refined_instance_planar_depth is not None and args.vis_refine:
                        refined_depth_error = np.abs(refined_instance_planar_depth - depth_gt)
                        valid_mask = depth_gt > 1e-4
                        refined_error_vis = vis_depth_error(refined_depth_error, valid_mask)
                        vis = np.hstack([vis, refined_error_vis])

                ori_ref_img = np.asarray(images['ori_ref_img'])[..., ::-1]
                ori_src_img = np.asarray(images['ori_src_img'])[..., ::-1]

                if args.save_pnt_cloud:
                    img_intrinsic = targets.get_field('intrinsic')
                    transform_pointcloud(ori_ref_img, instance_planar_depth, img_intrinsic, 'pnt_' + args.save_dir, images['ref_path'])

                vis = np.hstack([vis, ori_ref_img, ori_src_img])

                vis = cv2.resize(vis, None, fx=0.6, fy=0.6)

                if not osp.exists(args.save_dir):
                    os.makedirs(args.save_dir)

                scene_id, img_id = images['ref_path'].split('/')
                ref_img_id = img_id.split('.')[0]

                _, src_img_id = images['src_path'].split('/')
                cv2.imwrite(osp.join(args.save_dir, scene_id + '_' + ref_img_id + '_' + src_img_id), vis)

    if args.eval:
        if not args.eval_depth_only:
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

        if len(planar_depth_metrics) > 0:
            planar_depth_metrics = np.asarray(planar_depth_metrics)
            mean_planar_depth_metrics = np.mean(planar_depth_metrics, axis=0)
            mean_planar_depth_metrics = np.round(mean_planar_depth_metrics, 3)
            print('Mean Planar + Np Depth Metrics for Whole Img:', mean_planar_depth_metrics)

        if len(np_depth_metrics) > 0:
            np_depth_metrics = np.asarray(np_depth_metrics)
            mean_np_depth_metrics = np.mean(np_depth_metrics, axis=0)
            mean_np_depth_metrics = np.round(mean_np_depth_metrics, 3)
            print('Mean Non-Planar Depth Metrics:', mean_np_depth_metrics)

        img_depth_metrics = np.asarray(img_depth_metrics)
        mean_img_depth_metrics = np.mean(img_depth_metrics, axis=0)
        mean_img_depth_metrics = np.round(mean_img_depth_metrics, 3)
        print('Mean Planar Depth Metrics for Whole Img:', mean_img_depth_metrics)

        img_pixel_planar_depth_metrics = np.asarray(img_pixel_planar_depth_metrics)
        mean_img_pixel_planar_depth_metrics = np.mean(img_pixel_planar_depth_metrics, axis=0)
        mean_img_pixel_planar_depth_metrics = np.round(mean_img_pixel_planar_depth_metrics, 3)
        print('Mean Pixel-Planar Depth Metrics for Whole Img:', mean_img_pixel_planar_depth_metrics)

        if not args.eval_depth_only and len(gt_mask_img_depth_metrics) > 0:
            gt_mask_img_depth_metrics = np.asarray(gt_mask_img_depth_metrics)
            mean_gt_mask_img_depth_metrics = np.mean(gt_mask_img_depth_metrics, axis=0)
            mean_gt_mask_img_depth_metrics = np.round(mean_gt_mask_img_depth_metrics, 3)
            print('Mean Planar Depth Metric using gt masks for Whole Img:', mean_gt_mask_img_depth_metrics)

        planar_area_depth_metrics = np.asarray(planar_area_depth_metrics)
        mean_planar_area_depth_metrics = np.mean(planar_area_depth_metrics, axis=0)
        mean_planar_area_depth_metrics = np.round(mean_planar_area_depth_metrics, 3)
        print('Mean Pred-Planar-Area Depth Metrics:', mean_planar_area_depth_metrics)

        if len(planar_area_np_depth_metrics) > 0:
            planar_area_np_depth_metrics = np.asarray(planar_area_np_depth_metrics)
            mean_planar_area_np_depth_metrics = np.mean(planar_area_np_depth_metrics, axis=0)
            mean_planar_area_np_depth_metrics = np.round(mean_planar_area_np_depth_metrics, 3)
            print('Mean Pred-Planar-Area Np Depth Metrics:', mean_planar_area_np_depth_metrics)

        if not args.eval_depth_only:
            gt_planar_area_depth_metrics = np.asarray(gt_planar_area_depth_metrics)
            mean_gt_planar_area_depth_metrics = np.mean(gt_planar_area_depth_metrics, axis=0)
            mean_gt_planar_area_depth_metrics = np.round(mean_gt_planar_area_depth_metrics, 3)
            print('Mean Gt-Planar-Area Depth Metrics:', mean_gt_planar_area_depth_metrics)

            if len(refined_gt_planar_area_depth_metrics) > 0:
                refined_gt_planar_area_depth_metrics = np.asarray(refined_gt_planar_area_depth_metrics)
                mean_refined_gt_planar_area_depth_metrics = np.mean(refined_gt_planar_area_depth_metrics, axis=0)
                mean_refined_gt_planar_area_depth_metrics = np.round(mean_refined_gt_planar_area_depth_metrics, 3)
                print('Mean Refined Gt-Planar-Area Depth Metrics:', mean_refined_gt_planar_area_depth_metrics)

            if len(gt_planar_area_pixel_depth_metrics) > 0:
                gt_planar_area_pixel_depth_metrics = np.asarray(gt_planar_area_pixel_depth_metrics)
                mean_gt_planar_area_pixel_depth_metrics = np.mean(gt_planar_area_pixel_depth_metrics, axis=0)
                mean_gt_planar_area_pixel_depth_metrics = np.round(mean_gt_planar_area_pixel_depth_metrics, 3)
                print('Mean Gt-Planar-Area Pixel-Planar Depth Metrics:', mean_gt_planar_area_pixel_depth_metrics)

            if len(gt_planar_area_np_depth_metrics) > 0:
                gt_planar_area_np_depth_metrics = np.asarray(gt_planar_area_np_depth_metrics)
                mean_gt_planar_area_np_depth_metrics = np.mean(gt_planar_area_np_depth_metrics, axis=0)
                mean_gt_planar_area_np_depth_metrics = np.round(mean_gt_planar_area_np_depth_metrics, 3)
                print('Mean Gt-Planar-Area Np Depth Metrics:', mean_gt_planar_area_np_depth_metrics)

            label_mapping = targets.get_field('sem_mapping')
            n_class = len(label_mapping.keys()) + 1

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
    parser = argparse.ArgumentParser(description="Pytorch PlaneMVS Testing/Inference")

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

    parser.add_argument(
        '--eval_depth_only',
        help='Whether to only evaluate depth metrics and skip the plane detection metrics',
        action='store_true'
    )

    parser.add_argument(
        '--vis_refine',
        help='whether to visualize the refine depth and its error map',
        action='store_true'
    )

    args = parser.parse_args()

    inference(args)

    return


if __name__ == '__main__':
    main()
