import os
import os.path as osp

import cv2
import numpy as np

import tqdm


def load_depth(depth_path, divisor=1000.0, h=480, w=640):
    depth = cv2.imread(depth_path, -1)
    depth = cv2.resize(depth, (w, h))

    depth = depth / divisor

    return depth


def load_color(color_path, h=480, w=640):
    color = cv2.imread(color_path)
    color = cv2.resize(color, (w, h))

    return color


def load_intrinsic(intrinsic_path, h=480, w=640, ori_h=968, ori_w=1296):
    intrinsic = np.loadtxt(intrinsic_path)
    intrinsic = intrinsic[:3, :3]

    intrinsic[0][0] = intrinsic[0][0] * w / ori_w
    intrinsic[1][1] = intrinsic[1][1] * h / ori_h

    intrinsic[0][2] = w / 2
    intrinsic[1][2] = h / 2

    return intrinsic


def make_grid(intrinsic, depth, h=480, w=640):
    grid = np.ones((3, h, w))
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    grid[0, ...] = xs
    grid[1, ...] = ys

    grid = np.linalg.inv(intrinsic) @ grid.reshape(3, -1)
    grid = grid.reshape(3, h, w)

    points = grid * depth

    return grid, points


def fit_plane(points):
    # by this what we solve is n / d, where nx + d = 0
    plane = np.linalg.lstsq(points, -np.ones((points.shape[0], 1)), rcond=None)[0]
    plane = plane.squeeze()

    return plane


def depth_to_plane(points, depth, h=480, w=640, rel_depth_thresh=0.05):
    block_widths = np.asarray([1, 3, 6, 9])
    block_widths = np.concatenate([-block_widths, np.asarray([0]), block_widths])

    delta_xs, delta_ys = np.meshgrid(block_widths, block_widths)

    depth_valid_mask = depth > 1e-4
    valid_ys, valid_xs = np.where(depth_valid_mask)

    plane_map = np.zeros(points.shape)
    valid_mask = np.zeros(depth.shape).astype(np.bool)

    for c_x, c_y in zip(valid_xs, valid_ys):
        block_xs = c_x + delta_xs
        block_ys = c_y + delta_ys

        block_loc_valid_mask = (block_xs >= 0) * (block_ys >= 0) * (block_xs < w) * (block_ys < h)
        block_depth_valid_mask = depth[block_ys[block_loc_valid_mask], block_xs[block_loc_valid_mask]] > 1e-4

        valid_points = points[:, block_ys[block_loc_valid_mask][block_depth_valid_mask], block_xs[block_loc_valid_mask][block_depth_valid_mask]]

        dist_valid_mask = np.abs(valid_points[2, :] - depth[c_y, c_x]) < depth[c_y, c_x] * rel_depth_thresh

        valid_points = valid_points[:, dist_valid_mask].T

        if valid_points.shape[0] < 3:
            continue

        plane = fit_plane(valid_points)

        plane_map[:, c_y, c_x] = plane
        valid_mask[c_y, c_x] = True

    return plane_map, valid_mask


def load_data_paths(data_split_file):
    img_paths = []
    with open(data_split_file, 'r') as fp:
        for line in fp:
            img_paths.append(line.strip().split('\t')[0])

    return img_paths


def vis_plane_map(plane_map, color, valid_mask, save_id):
    min_val = np.min(plane_map)
    plane_map = plane_map - min_val

    max_val = np.max(plane_map)

    normalized_plane_map = plane_map / max_val
    normalized_plane_map = 1 - normalized_plane_map

    normalized_plane_map[:, ~valid_mask] = 0

    vis = (normalized_plane_map * 255).transpose(1, 2, 0)

    vis = cv2.resize(vis, (640, 480))
    vis = np.hstack([vis, color])

    save_dir = 'debug_plane_vis'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    cv2.imwrite(osp.join(save_dir, save_id), vis)


def main(data_split_file, data_root, save_dir, vis=False):
    img_paths = load_data_paths(data_split_file)

    h, w = 120, 160

    for img_path in tqdm.tqdm(img_paths):
        scene_id, img_id = img_path.split('/')

        img_path = osp.join(data_root, scene_id, 'frames/color', img_id)
        depth_path = img_path.replace('color', 'depth').replace('jpg', 'png')
        intrinsic_path = osp.join(data_root, scene_id, 'frames/intrinsic/intrinsic_color.txt')

        depth = load_depth(depth_path, h=h, w=w)
        intrinsic = load_intrinsic(intrinsic_path, h=h, w=w)

        grid, points = make_grid(intrinsic, depth, h=h, w=w)

        plane_map, valid_mask = depth_to_plane(points, depth, h=h, w=w)

        if vis:
            color = load_color(img_path, h=480, w=640)
            save_id = scene_id + '_' + img_id + '.png'

            vis_plane_map(plane_map, color, valid_mask, save_id)

        plane_map = plane_map.transpose(1, 2, 0)

        # plane_map = cv2.resize(plane_map.astype(np.float64), (640, 480), cv2.INTER_LINEAR)
        # valid_mask = cv2.resize(valid_mask.astype(np.float64), (640, 480), cv2.INTER_LINEAR)

        valid_mask = valid_mask > 0.5

        plane_save_id = scene_id + '_' + img_id.split('.')[0] + '_plane' + '.npy'
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        np.save(osp.join(save_dir, plane_save_id), plane_map)

        mask_save_id = scene_id + '_' + img_id.split('.')[0] + '_mask' + '.npy'

        np.save(osp.join(save_dir, mask_save_id), valid_mask)

if __name__ == '__main__':
    train_split_file = '../data_splits/valid_stereo_val_files.txt'
    data_root = '/mnt/Data/jiachenliu/scannet_data/scans'

    save_dir = 'sampled_pixel_planar_map'

    main(train_split_file, data_root, save_dir)
