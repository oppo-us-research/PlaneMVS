import os
import os.path as osp

import numpy as np

import cv2


intrinsic = np.asarray([
    [577.87, 0, 320],
    [0, 577.87, 240],
    [0, 0, 1]
])


def get_grid(h=480, w=640):
    xxs, yys = np.meshgrid(np.arange(w), np.arange(h))
    xys = np.ones((3, h, w))

    xys[0, ...] = xxs
    xys[1, ...] = yys

    grids = np.linalg.inv(intrinsic) @ xys.reshape(3, -1)
    grids = grids.reshape(3, h, w)

    return grids


def plane_to_depth(grids, planes, masks, h=480, w=640):
    offsets = -planes[:, -1]
    normals = planes[:, :3]

    grids = grids.reshape(3, -1)

    piece_depths = offsets[..., None] / ((normals @ grids) + 1e-10)

    plane_num = piece_depths.shape[0]
    piece_depths = piece_depths.reshape(plane_num, h, w) * masks

    return piece_depths


def clean_segmentation(image, grids, planes, masks,
                       plane_ids,
                       gt_depth,
                       planeAreaThreshold=200,
                       planeWidthThreshold=10,
                       depthDiffThreshold=0.1,
                       validAreaThreshold=0.5,
                       brightThreshold=20):
    plane_depths = plane_to_depth(grids, planes, masks)

    img_mask = np.logical_and(np.linalg.norm(image, axis=-1) > brightThreshold, gt_depth > 1e-4)
    depth_diff_mask = np.logical_or(np.abs(plane_depths - gt_depth) < depthDiffThreshold, gt_depth < 1e-4)

    new_segmentation = np.full(img_mask.shape, fill_value=-1)
    new_masks = []
    new_plane_depths = []
    new_planes = []

    assert len(plane_ids) == len(masks) == len(plane_depths)

    for idx, (plane_id, mask) in enumerate(zip(plane_ids, masks)):
        ori_area = mask.sum()

        valid_mask = np.logical_and(mask, depth_diff_mask[idx])
        new_area = np.logical_and(valid_mask, img_mask).sum()

        if new_area < ori_area * validAreaThreshold:
            continue

        valid_mask = valid_mask.astype(np.uint8)
        valid_mask = cv2.dilate(valid_mask, np.ones((3, 3)))

        num_labels, components = cv2.connectedComponents(valid_mask)

        for label in range(1, num_labels):
            mask = components == label

            ys, xs = mask.nonzero()
            area = float(len(xs))

            if area < planeAreaThreshold:
                continue

            size_y = ys.max() - ys.min() + 1
            size_x = xs.max() - xs.min() + 1
            length = np.linalg.norm([size_x, size_y])

            if area / length < planeWidthThreshold:
                continue

            new_segmentation[mask] = plane_id

        new_masks.append(new_segmentation == plane_id)
        new_plane_depths.append(plane_depths[idx])
        new_planes.append(planes[idx])

    if len(new_masks) == 0:
        return None, None, None

    new_segmentation = new_segmentation.astype(np.uint16)

    new_masks = np.asarray(new_masks)
    new_plane_depths = np.asarray(new_plane_depths)
    new_planes = np.asarray(new_planes)

    img_plane_depth = np.sum(new_plane_depths * new_masks, axis=0)
    img_plane_mask = new_masks.max(0) * (gt_depth > 1e-4)
    img_plane_area = img_plane_mask.sum()

    depth_error = (np.abs(img_plane_depth - gt_depth) * img_plane_mask).sum() / (img_plane_area + 1e-4)

    return new_segmentation, depth_error, new_planes
