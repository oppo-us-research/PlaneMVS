import os
import os.path as osp

import cv2
import tqdm
import numpy as np

from clean_seg import get_grid, clean_segmentation
from cluster import cluster

data_dir = '/data/panji/ScanNet/scans'
save_dir = 'cleaned_segmentation'
train_split_file = 'scannet_splits/train_files.txt'

np.random.seed(428)

depth_error_thresh = 0.1


def load_image(image_path, resize=(640, 480)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, resize)

    return img


def load_pose(pose_path):
    return np.loadtxt(pose_path)


def load_raw_seg(seg_path):
    raw_seg = cv2.imread(seg_path, -1).astype(np.int32)

    return raw_seg


def load_depth(depth_path, divisor=1000.0):
    depth = cv2.imread(depth_path, -1) / divisor

    return depth


def raw_seg_to_seg(raw_seg):
    seg = raw_seg[..., 2] * 256 * 256 + raw_seg[..., 1] * 256 + raw_seg[..., 0]
    seg = seg // 100 - 1

    return seg


def filter_seg(seg):
    valid_ids = []
    masks = []

    for element in np.unique(seg):
        if element != -1 and element != 167771:
            valid_ids.append(element)
            masks.append(seg == element)

    return valid_ids, masks


def load_planes(plane_path):
    planes = np.load(plane_path)

    return planes


def transform_planes(raw_planes, valid_ids, pose):
    valid_planes = raw_planes[valid_ids]

    real_d = np.linalg.norm(valid_planes, axis=-1) + 1e-10
    real_d = real_d[:, None]

    real_n = valid_planes / real_d

    world_planes = np.concatenate([real_n, -real_d], axis=-1)
    camera_planes = world_planes @ pose

    return camera_planes


def load_data_paths(split_file):
    data_paths = []

    with open(split_file, 'r') as fp:
        for line in fp:
            scene, img_id = line.strip().split(' ')[:2]
            data_paths.append(osp.join(scene, img_id + '.jpg'))

    return data_paths


def main(do_cluster=False):
    grids = get_grid()
    data_paths = load_data_paths(train_split_file)

    all_camera_normals = []

    valid_paths = []

    for rel_img_path in tqdm.tqdm(data_paths):
        scene, img_id = rel_img_path.split('/')

        if osp.exists(osp.join(save_dir, scene, img_id.replace('jpg', 'png'))):
            continue

        img_dir = osp.join(data_dir, scene, 'frames/color')
        img_path = osp.join(img_dir, img_id)
        if not osp.exists(img_path):
            continue

        seg_path = img_path.replace('frames/color', 'annotation/segmentation').replace('jpg', 'png')
        depth_path = img_path.replace('color', 'depth').replace('jpg', 'png')
        pose_path = img_path.replace('color', 'pose').replace('jpg', 'txt')
        plane_path = osp.join(data_dir, scene, 'annotation/planes.npy')

        image = load_image(img_path)
        gt_depth = load_depth(depth_path)

        raw_seg = load_raw_seg(seg_path)
        seg = raw_seg_to_seg(raw_seg)
        valid_ids, masks = filter_seg(seg)

        if len(valid_ids) == 0:
            continue

        pose = load_pose(pose_path)
        planes = load_planes(plane_path)
        camera_planes = transform_planes(planes, valid_ids, pose)

        new_segmentation, depth_error, new_planes = clean_segmentation(image, grids, camera_planes, masks, valid_ids, gt_depth)

        if new_segmentation is None:
            continue

        # bad geometry align
        if depth_error > depth_error_thresh:
            continue

        # No plane instances
        if len(np.unique(new_segmentation)) == 1 and 65535 in np.unique(new_segmentation):
            continue

        save_folder = osp.join(save_dir, scene)
        if not osp.exists(save_folder):
            os.makedirs(save_folder)

        cv2.imwrite(osp.join(save_folder, img_id.replace('jpg', 'png')), new_segmentation)
        all_camera_normals.append(new_planes[:, :3])

        valid_paths.append(rel_img_path)

    with open('train_files.txt', 'w') as fp:
        for path in valid_paths:
            fp.write(path)
            fp.write('\n')

    if do_cluster:
        all_camera_normals = np.concatenate(all_camera_normals, axis=0)
        anchor_normals = cluster(all_camera_normals, cluster_anchor_num=7)

        np.save('anchor_normals.npy', anchor_normals)

        print(anchor_normals)
        exit(1)

if __name__ == '__main__':
    main(do_cluster=False)
