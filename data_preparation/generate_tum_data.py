import os
import os.path as osp

import glob
import tqdm

import numpy as np

np.random.seed(428)


def get_all_tum_path(data_root, sample=-1):
    # the near scenes contained in ORB-SLAM3
    scene_exts = ['freiburg1_desk', 'freiburg1_room', 'freiburg1_desk2', 'freiburg2_desk', 'freiburg3_long_office_household', 'freiburg3_long_office_household_validation']

    all_depths = []
    all_images = []
    all_poses = []

    all_depth_to_pose_mappings = []
    all_depth_to_rgb_mappings = []

    for scene_idx, scene_ext in enumerate(scene_exts):
        scene = 'rgbd_dataset_' + scene_ext

        pose_timestamps, poses = read_pose_mapping(data_root, scene)
        image_timestamps, image_filenames = read_rgb_mapping(data_root, scene)

        depth_timestamps, depth_filenames = read_depth_mapping(data_root, scene)

        depth_to_pose_mapping = get_closet_matches(depth_timestamps, pose_timestamps)
        depth_to_rgb_mapping = get_closet_matches(depth_timestamps, image_timestamps)

        all_depths.append(depth_filenames)
        all_images.append(image_filenames)
        all_poses.append(poses)

        all_depth_to_pose_mappings.append(depth_to_pose_mapping)
        all_depth_to_rgb_mappings.append(depth_to_rgb_mapping)

    return all_depths, all_images, all_poses, all_depth_to_pose_mappings, all_depth_to_rgb_mappings


def get_rel_trans(ref_pose, src_pose):
    rel_pose = np.linalg.inv(src_pose) @ ref_pose
    rel_trans = np.linalg.norm(rel_pose[:3, -1])

    return rel_trans


def read_rgb_mapping(data_root, scene):
    map_file = osp.join(data_root, scene, 'rgb.txt')

    image_timestamps = np.loadtxt(map_file, usecols=0)
    image_filenames = sorted(glob.glob(osp.join(data_root, scene, "rgb", "*.png")))

    return image_timestamps, image_filenames


def read_pose_mapping(data_root, scene):
    seq_file = osp.join(data_root, scene, 'groundtruth.txt')
    poses_with_quat = np.loadtxt(seq_file)

    pose_timestamps = poses_with_quat[:, 0]
    poses = poses_with_quat[:, 1:]

    return pose_timestamps, poses


def quar_to_pose_matrix(pose_vec):
    pose_location = pose_vec[:3]
    pose_quaternion = pose_vec[3:]
    from scipy.spatial.transform import Rotation

    rot_scipy = Rotation.from_quat(pose_quaternion).as_matrix()
    trans_vec = np.asarray(pose_location).T

    pose_matrix = np.eye(4)

    pose_matrix[:3, :3] = rot_scipy
    pose_matrix[:3, 3] = trans_vec

    return pose_matrix


def get_closet_matches(tgt_timestamps, other_timestamps):
    differences = np.abs(other_timestamps[..., None] - tgt_timestamps[None, ...])
    indexes = np.argmin(differences, axis=0)

    return indexes


def read_depth_mapping(data_root, scene):
    depth_timestamps = np.loadtxt(osp.join(data_root, scene, "depth.txt"), usecols=0)
    depth_filenames = sorted(glob.glob(osp.join(data_root, scene, "depth", "*.png")))

    return depth_timestamps, depth_filenames


def generate_stereo_paths(all_depth_paths,
                          all_color_paths,
                          all_poses,
                          all_depth_to_pose_mappings,
                          all_depth_to_rgb_mappings,
                          interval_range=20,
                          verify_trans=True,
                          lower_thresh=0.05,
                          higher_thresh=0.15,
                          save_pose=True):
    stereo_pair_paths = []
    pose_pairs = []
    depth_pairs = []

    for scene_idx, scene_depth_paths in enumerate(all_depth_paths):
        scene_stereo_pair_paths = []
        scene_pose_pairs = []
        scene_depth_pairs = []

        scene_depth_to_pose = all_depth_to_pose_mappings[scene_idx]
        scene_depth_to_rgb = all_depth_to_rgb_mappings[scene_idx]

        scene_image_paths = all_color_paths[scene_idx]
        scene_poses = all_poses[scene_idx]

        for depth_idx, ref_depth_path in enumerate(tqdm.tqdm(scene_depth_paths)):
            src_valid = False
            max_iter = 50
            start_iter = 0

            color_idx = scene_depth_to_rgb[depth_idx]
            pose_idx = scene_depth_to_pose[depth_idx]

            ref_color_path = scene_image_paths[color_idx]
            ref_pose = scene_poses[pose_idx]

            ref_pose = quar_to_pose_matrix(ref_pose)
            scene_name = ref_color_path.split('/')[-3]

            while not src_valid:
                start_iter += 1
                interval = np.random.choice(np.arange(-interval_range, interval_range), size=1, replace=False)[0]

                if interval == 0:
                    continue

                src_depth_idx = depth_idx + interval
                if src_depth_idx < 0 or src_depth_idx >= len(scene_depth_paths):
                    continue

                src_depth_path = scene_depth_paths[src_depth_idx]

                src_img_idx = scene_depth_to_rgb[src_depth_idx]
                src_color_path = scene_image_paths[src_img_idx]

                src_pose_idx = scene_depth_to_pose[src_depth_idx]
                src_pose = scene_poses[src_pose_idx]

                src_pose = quar_to_pose_matrix(src_pose)

                if np.any(np.isnan(ref_pose)) or np.any(np.isinf(ref_pose)) or np.any(np.isneginf(ref_pose)):
                    break

                if np.any(np.isnan(src_pose)) or np.any(np.isinf(src_pose)) or np.any(np.isneginf(src_pose)):
                    continue

                rel_trans = get_rel_trans(ref_pose, src_pose)

                if rel_trans >= lower_thresh and rel_trans <= higher_thresh:
                    src_valid = True

                if start_iter > max_iter:
                    break

            if src_valid:
                src_path = osp.join(scene_name, 'rgb', src_color_path.split('/')[-1])
                ref_path = osp.join(scene_name, 'rgb', ref_color_path.split('/')[-1])

                src_depth_path = osp.join(scene_name, 'depth', src_depth_path.split('/')[-1])
                ref_depth_path = osp.join(scene_name, 'depth', ref_depth_path.split('/')[-1])

                scene_stereo_pair_paths.append([ref_path, src_path])
                scene_pose_pairs.append([ref_pose, src_pose])
                scene_depth_pairs.append([ref_depth_path, src_depth_path])

        if save_pose:
            save_dir = osp.join('tumrgbd_stereo_poses', scene_name)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)

            for idx, pair in enumerate(tqdm.tqdm(scene_stereo_pair_paths)):
                pose_pair = scene_pose_pairs[idx]

                ref_path, src_path = pair
                ref_pose, src_pose = pose_pair

                if not osp.exists(osp.join(save_dir, ref_path.replace('png', 'txt'))):
                    np.savetxt(osp.join(save_dir, ref_path.split('/')[-1].replace('png', 'txt')), ref_pose)

                if not osp.exists(osp.join(save_dir, src_path.replace('png', 'txt'))):
                    np.savetxt(osp.join(save_dir, src_path.split('/')[-1].replace('png', 'txt')), src_pose)

        stereo_pair_paths += scene_stereo_pair_paths
        pose_pairs += scene_pose_pairs
        depth_pairs += scene_depth_pairs

    with open('tumrgbd_stereo_files.txt', 'w') as fp:
        for idx, pair in enumerate(stereo_pair_paths):
            ref_path, src_path = pair
            ref_depth_path, src_depth_path = depth_pairs[idx]

            fp.write(ref_path)
            fp.write('\t')
            fp.write(ref_depth_path)
            fp.write('\t')
            fp.write(src_path)
            fp.write('\t')
            fp.write(src_depth_path)
            fp.write('\n')


def generate_stereo_paths_with_dvmvs_poses(all_depth_paths,
                                           all_color_paths,
                                           all_depth_to_rgb_mappings,
                                           interval_range=20,
                                           lower_thresh=0.05,
                                           higher_thresh=0.15,
                                           dvmvs_pose_dir=None):
    stereo_pair_paths = []
    depth_pairs = []

    for scene_idx, scene_depth_paths in enumerate(all_depth_paths):
        scene_stereo_pair_paths = []
        scene_depth_pairs = []

        scene_depth_to_rgb = all_depth_to_rgb_mappings[scene_idx]

        scene_image_paths = all_color_paths[scene_idx]

        for depth_idx, ref_depth_path in enumerate(tqdm.tqdm(scene_depth_paths)):
            src_valid = False
            max_iter = 50

            start_iter = 0
            color_idx = scene_depth_to_rgb[depth_idx]

            ref_color_path = scene_image_paths[color_idx]

            scene_name = ref_color_path.split('/')[-3]

            ref_pose_path = osp.join(dvmvs_pose_dir, scene_name, ref_color_path.split('/')[-1].replace('png', 'txt'))
            ref_pose = np.loadtxt(ref_pose_path)

            while not src_valid:
                start_iter += 1
                interval = np.random.choice(np.arange(-interval_range, interval_range), size=1, replace=False)[0]

                if interval == 0:
                    continue

                src_depth_idx = depth_idx + interval
                if src_depth_idx < 0 or src_depth_idx >= len(scene_depth_paths):
                    continue

                src_depth_path = scene_depth_paths[src_depth_idx]

                src_img_idx = scene_depth_to_rgb[src_depth_idx]
                src_color_path = scene_image_paths[src_img_idx]

                src_pose_path = osp.join(dvmvs_pose_dir, scene_name, src_color_path.split('/')[-1].replace('png', 'txt'))
                src_pose = np.loadtxt(src_pose_path)

                rel_trans = get_rel_trans(ref_pose, src_pose)

                if rel_trans >= lower_thresh and rel_trans <= higher_thresh:
                    src_valid = True

                if start_iter > max_iter:
                    break

            if src_valid:
                ref_path = osp.join(scene_name, 'rgb', ref_color_path.split('/')[-1])
                src_path = osp.join(scene_name, 'rgb', src_color_path.split('/')[-1])

                ref_depth_path = osp.join(scene_name, 'depth', ref_depth_path.split('/')[-1])
                src_depth_path = osp.join(scene_name, 'depth', src_depth_path.split('/')[-1])

                scene_stereo_pair_paths.append([ref_path, src_path])
                scene_depth_pairs.append([ref_depth_path, src_depth_path])

        stereo_pair_paths += scene_stereo_pair_paths
        depth_pairs += scene_depth_pairs

    with open('dvmvs_tumrgbd_stereo_files.txt', 'w') as fp:
        for idx, pair in enumerate(stereo_pair_paths):
            ref_path, src_path = pair
            ref_depth_path, src_depth_path = depth_pairs[idx]

            fp.write(ref_path)
            fp.write('\t')
            fp.write(ref_depth_path)
            fp.write('\t')
            fp.write(src_path)
            fp.write('\t')
            fp.write(src_depth_path)
            fp.write('\n')


def main(data_root):
    all_depth_paths, all_image_paths, all_poses, all_depth_to_pose_mappings, all_depth_to_rgb_mappings = get_all_tum_path(data_root)

    pose_dir = '/data/home/uss00026/deep-video-mvs/dataset/tum-rgbd-export/dvmvs_pose_outputs'
    generate_stereo_paths_with_dvmvs_poses(all_depth_paths, all_image_paths, all_depth_to_rgb_mappings, dvmvs_pose_dir=pose_dir)


if __name__ == '__main__':
    data_root = '/mnt/Data/panji/TUM-RGBD'
    main(data_root)
