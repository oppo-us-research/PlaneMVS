import os
import os.path as osp

import cv2
import tqdm
import numpy as np

from glob import glob

from .clean_seg import get_grid, clean_segmentation



# load image and resize
def load_image(image_path, resize=(640, 480)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, resize)

    return img


# load ctw pose
def load_pose(pose_path):
    return np.loadtxt(pose_path)


# load raw segmentation
def load_raw_seg(seg_path):
    raw_seg = cv2.imread(seg_path, -1).astype(np.int32)

    return raw_seg


# load depth and turn into meter
def load_depth(depth_path, divisor=1000.0):
    depth = cv2.imread(depth_path, -1) / divisor

    return depth


# transfer raw (3-channel) seg map to 1-channel map
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


# load raw plane params, which is from world coord
def load_planes(plane_path):
    planes = np.load(plane_path)

    return planes


# transform from world coord to camera coord
def transform_planes(raw_planes, valid_ids, pose):
    valid_planes = raw_planes[valid_ids]

    real_d = np.linalg.norm(valid_planes, axis=-1) + 1e-10
    real_d = real_d[:, None]

    real_n = valid_planes / real_d

    world_planes = np.concatenate([real_n, -real_d], axis=-1)
    camera_planes = world_planes @ pose

    return camera_planes


def sample_imgs(sample_num=100000):
    with open(scene_split_file, 'r') as fp:
        scenes = [l.strip() for l in fp]

    all_img_paths = []
    for scene in scenes:
        img_paths = glob(osp.join(data_dir, f'{scene}/frames/color/*.jpg'))

        if len(img_paths) > 100:
            img_paths = np.random.choice(img_paths, 100, replace=False)

        all_img_paths.extend(img_paths)

    all_img_paths = all_img_paths[:sample_num]

    return all_img_paths


def generate_stereo_paths(interval_range=20, verify_trans=True, lower_thresh=0.05, higher_thresh=0.15):
    stereo_pair_paths = []

    img_paths = sample_imgs(sample_num=100)

    for img_path in img_paths:

        scene = img_path.split('/')[-4]
        ref_img_id = img_path.split('/')[-1]
        ref_img_id = ref_img_id.split('.')[0]

        src_valid = False

        # if after max_iter, we still cannot match a valid source img, skip this ref img
        max_iter = 50
        start_iter = 0
        while not src_valid:
            start_iter += 1
            interval = np.random.choice(np.arange(-interval_range, interval_range), size=1, replace=False)[0]

            if interval == 0:
                continue

            src_img_id = str(int(ref_img_id) + interval)

            if osp.exists(osp.join(data_dir, scene, 'frames/color', src_img_id + '.jpg')):
                ref_path = osp.join(scene, ref_img_id + '.jpg')
                src_path = osp.join(scene, src_img_id + '.jpg')

                if verify_trans:
                    ref_pose_path = osp.join(data_dir, scene, 'frames/pose', ref_img_id + '.txt')
                    src_pose_path = osp.join(data_dir, scene, 'frames/pose', src_img_id + '.txt')

                    # make sure the pose is valid
                    ref_pose = load_pose(ref_pose_path)
                    if np.any(np.isnan(ref_pose)) or np.any(np.isinf(ref_pose)) or np.any(np.isneginf(ref_pose)):
                        break

                    src_pose = load_pose(src_pose_path)
                    if np.any(np.isnan(src_pose)) or np.any(np.isinf(src_pose)) or np.any(np.isneginf(src_pose)):
                        continue

                    # relative translation
                    rel_trans = get_rel_trans(ref_pose, src_pose)

                    # make sure the two views are not too close or too distant
                    if rel_trans < lower_thresh or rel_trans > higher_thresh:
                        src_valid = False

                    else:
                        src_valid = True

            if start_iter > max_iter:
                break

        # if finally the pair is qualified, we save it into the list
        if src_valid:
            stereo_pair_paths.append([ref_path, src_path])

    if len(stereo_pair_paths) > 0:
        with open(stereo_split_file, 'w') as fp:
            for pair_path in stereo_pair_paths:
                ref_path, src_path = pair_path

                fp.write(ref_path)
                fp.write('\t')
                fp.write(src_path)
                fp.write('\n')

    return stereo_split_file


def load_stereo_paths(stereo_split_file):
    data_paths = []

    with open(stereo_split_file, 'r') as fp:
        for line in fp:
            ref_path, src_path = line.strip().split('\t')

            data_paths.append([ref_path, src_path])

    return data_paths


def get_rel_trans(ref_pose, src_pose):
    rel_pose = np.linalg.inv(src_pose) @ ref_pose
    rel_trans = np.linalg.norm(rel_pose[:3, -1])

    return rel_trans


# the main pre-processing flow
def main():
    grids = get_grid()
    generate_stereo = not osp.exists(stereo_split_file)
    if generate_stereo:
        generate_stereo_paths()

    print('pair generation done, start cleaning...')

    data_paths = load_stereo_paths(stereo_split_file)

    valid_paths = []

    for ref_rel_img_path, src_rel_img_path in tqdm.tqdm(data_paths):
        scene, ref_img_id = ref_rel_img_path.split('/')
        scene, src_img_id = src_rel_img_path.split('/')

        img_dir = osp.join(data_dir, scene, 'frames/color')
        ref_img_path = osp.join(img_dir, ref_img_id)
        if not osp.exists(ref_img_path):
            continue

        src_img_path = osp.join(img_dir, src_img_id)
        if not osp.exists(src_img_path):
            continue

        ref_valid = False
        src_valid = False

        for idx, img_path in enumerate([ref_img_path, src_img_path]):
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
            if depth_error > 0.1:
                break

            # No plane instances
            if len(np.unique(new_segmentation)) == 1 and 65535 in np.unique(new_segmentation):
                break

            # record whether ref img is valid
            if idx == 0:
                ref_valid = True
                ref_save = new_segmentation

            # record whether src img is valid
            else:
                src_valid = True
                src_save = new_segmentation

        if ref_valid and src_valid:
            save_folder = osp.join(cleaned_seg_save_dir, scene)
            if not osp.exists(save_folder):
                os.makedirs(save_folder)

            cv2.imwrite(osp.join(save_folder, ref_img_path.split('/')[-1].replace('jpg', 'png')), ref_save)
            cv2.imwrite(osp.join(save_folder, src_img_path.split('/')[-1].replace('jpg', 'png')), src_save)

            valid_paths.append([ref_rel_img_path, src_rel_img_path])

    with open(cleaned_stereo_split_file, 'w') as fp:
        for path in valid_paths:
            ref_path, src_path = path
            fp.write(ref_path)
            fp.write('\t')
            fp.write(src_path)

            fp.write('\n')


if __name__ == '__main__':
    # mount your ScanNet dataset to the local dir "./datasets/scannet_data"
    data_dir = './datasets/scannet_data/scans'
    cleaned_seg_save_dir = 'stereo_cleaned_segmentation'

    split = 'train'
    scene_split_file = osp.join(osp.dirname(data_dir), 'ScanNet/Tasks/Benchmark/scannetv2_{}.txt').format(split)

    os.makedirs('scannet_splits', exist_ok=True)

    stereo_split_file = 'scannet_splits/scannet_stereo_{}_files.txt'.format(split)
    cleaned_stereo_split_file = 'scannet_splits/cleaned_scannet_stereo_{}_files.txt'.format(split)

    np.random.seed(428)

    main()
