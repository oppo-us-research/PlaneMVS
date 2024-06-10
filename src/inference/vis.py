"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import os.path as osp

import cv2

import matplotlib as mpl
import matplotlib.cm as cm

import numpy as np
import torch

from PIL import Image

""" load our own modules """
from src.utils.colormap import ColorPalette



def vis_det(pred, origin_img, sem_mapping, alpha=0.5, use_src=False, vis_bbox=False):
    colors = ColorPalette(100).getColorMap(returnTuples=True)

    img = np.asarray(origin_img)[..., ::-1]

    if pred is None:
        return img * alpha

    pred = [p.to('cpu') for p in pred]

    assert len(pred) == 1, "Currently only support batch_size=1 visualization"
    pred = pred[0]

    if pred.has_field('final_scores'):
        scores = pred.get_field('final_scores').numpy()
        plane_scores = pred.get_field('plane_scores').numpy()

    else:
        scores = pred.get_field('scores').numpy()
        plane_scores = None

    rank = scores.argsort()
    scores = scores[rank]
    labels = pred.get_field('labels').numpy()[rank]

    #bboxes = pred.bbox.numpy().astype(np.int)[rank] #np.int was deprecated in NumPy 1.20 and was removed in NumPy 1.24;
    bboxes = pred.bbox.numpy().astype(np.int32)[rank]
    masks = pred.get_field('mask').numpy()[rank]

    if masks.ndim == 4:
        masks = masks.squeeze(axis=1)

    if pred.has_field('refined_mask'):
        refined_masks = pred.get_field('refined_mask').squeeze(dim=1).numpy()[rank]

    else:
        refined_masks = None

    ins_res = np.zeros(img.shape)

    for idx, mask in enumerate(masks):
        color = colors[idx]
        ins_res[mask > 0, :] = color

    ins_res = img * alpha + ins_res * (1 - alpha)

    if refined_masks is not None:
        refined_ins_res = np.zeros(img.shape)

        for idx, refined_mask in enumerate(refined_masks):
            color = colors[idx]
            refined_ins_res[refined_mask > 0, :] = color

        refined_ins_res = img * alpha + refined_ins_res * (1 - alpha)

        ins_res = np.hstack([ins_res, refined_ins_res])

    sem_res = np.zeros(img.shape)

    for label, mask in zip(labels, masks):
        color = colors[label]
        sem_res[mask > 0, :] = color

    sem_res = img * alpha + sem_res * (1 - alpha)

    if vis_bbox:
        if plane_scores is not None:
            label_tmp = '{}: {:.2f}, {:.2f}'

        else:
            label_tmp = '{}: {:.2f}'

        for idx, (box, label, score) in enumerate(zip(bboxes, labels, scores)):
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()

            color = colors[label]
            cv2.rectangle(ins_res, tuple(top_left), tuple(bottom_right), tuple(color), 1)

            label_txt = sem_mapping[label]

            if plane_scores is not None:
                txt = label_tmp.format(label_txt, score, plane_scores[idx])

            else:
                txt = label_tmp.format(label_txt, score)

            cv2.putText(
                ins_res, txt, (top_left[0] + 5, top_left[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(color), 2)

    vis = np.hstack([ins_res, sem_res])

    return vis


def vis_gt_det(target, origin_img, gt_planar_semantic_map, alpha=0.5):
    colors = ColorPalette(100).getColorMap(returnTuples=True)

    img = np.asarray(origin_img)[..., ::-1]

    target_masks = target.get_field('masks').instances.masks
    target_masks = target_masks.cpu().numpy()

    res = np.zeros(img.shape)

    for idx, mask in enumerate(target_masks):
        color = colors[idx]
        res[mask > 0, :] = color

    res = img * alpha + res * (1 - alpha)

    sem_res = np.zeros(img.shape)

    for label in np.unique(gt_planar_semantic_map):
        if label == 0:
            continue

        color = colors[label]

        sem_mask = gt_planar_semantic_map == label
        sem_res[sem_mask, :] = color

    sem_res = img * alpha + sem_res * (1 - alpha)
    res = np.hstack([res, sem_res])

    return res


def depth_to_disp(depth, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth

    disp = 1 / (depth + 1e-4)
    scaled_disp = min_disp + (max_disp - min_disp) * disp

    return disp, scaled_disp, min_disp, max_disp


def vis_disparity(depth, min_depth=0.1, max_depth=5):
    disp, scaled_disp, min_disp, max_disp = depth_to_disp(depth, min_depth=min_depth, max_depth=max_depth)

    vmax = np.percentile(disp[depth > 1e-4], 95)
    normalizer = mpl.colors.Normalize(vmin=min_disp, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')

    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[depth < 1e-4, :] = 0

    im = Image.fromarray(colormapped_im)

    return im


def vis_depth(depth, min_depth=0, max_depth=5):
    normalizer = mpl.colors.Normalize(vmin=min_depth, vmax=max_depth)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')

    colormapped_im = (mapper.to_rgba(depth)[..., :3] * 255).astype(np.uint8)
    im = Image.fromarray(colormapped_im)

    return im


def vis_depth_error(depth_error, valid_mask, max_error=1):
    depth_error = depth_error.clip(max=max_error)
    gray_error_map = np.stack([depth_error * 255, depth_error * 255, depth_error * 255], axis=-1)
    gray_error_map[~valid_mask, :] = 0

    return gray_error_map


def depth_to_points(intrinsic, depth, h=480, w=640):
    # indexing added for Torch >=1.10;
    # ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    intrinsic = intrinsic.cpu().numpy().astype(float)

    grids = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
    grids = grids.transpose(2, 0, 1)

    depth = depth.astype(float)

    camera_grid = np.linalg.inv(intrinsic[:3, :3]) @ grids.reshape(3, -1)
    camera_grid = camera_grid.reshape(3, h, w)

    points = camera_grid * depth

    return points


def transform_pointcloud(rgb_img, planar_depth, img_intrinsic, save_dir, img_path):
    assert rgb_img.shape[:2] == planar_depth.shape

    pred_xyz = depth_to_points(img_intrinsic, planar_depth)
    pred_xyz = pred_xyz.transpose(1, 2, 0)

    point_with_rgb = np.concatenate([pred_xyz, rgb_img], axis=-1)
    point_with_rgb = point_with_rgb[10:-10, 10:-10, ...]
    point_with_rgb = point_with_rgb.reshape(-1, 6)

    valid_mask = point_with_rgb[..., 2] > 0
    point_with_rgb = point_with_rgb[valid_mask]

    points_to_write = []

    for point in point_with_rgb:
        x, y, z, b, g, r = point
        points_to_write.append('%f %f %f %d %d %d 0\n' % (x,y,z,r,g,b))

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    scene_id, img_id = img_path.split('/')
    save_path = (osp.join(save_dir, scene_id + '_' + img_id)).replace('jpg', 'ply')

    write_pointcloud(save_path, points_to_write)
    # exit(1)


def write_pointcloud(save_path, points):
    with open(save_path,'w') as fp:
        fp.write("""ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
        """ % (len(points),"".join(points)))

    return
