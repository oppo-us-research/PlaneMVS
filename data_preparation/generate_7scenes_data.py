import os
import os.path as osp

import glob
import tqdm

import numpy as np

np.random.seed(428)

train_seqs = ['chess/seq-01', 'chess/seq-02', 'chess/seq-04', 'chess/seq-06',
              'fire/seq-01', 'fire/seq-02',
              'heads/seq-02',
              'office/seq-01', 'office/seq-03', 'office/seq-04', 'office/seq-05', 'office/seq-06', 'office/seq-08', 'office/seq-10',
              'pumpkin/seq-02', 'pumpkin/seq-03', 'pumpkin/seq-06', 'pumpkin/seq-08',
              'redkitchen/seq-01', 'redkitchen/seq-02', 'redkitchen/seq-05', 'redkitchen/seq-07', 'redkitchen/seq-08', 'redkitchen/seq-11', 'redkitchen/seq-13',
              'stairs/seq-02', 'stairs/seq-03', 'stairs/seq-05', 'stairs/seq-06']


def get_all_7scenes_path(data_root, sample=-1):
    all_colors = glob.glob(osp.join(data_root, '*', 'seq*', '*color.png'))

    # if sample > 0:
    #     sampled_colors = np.random.choice(all_colors, size=sample, replace=False)

    # else:
    #     sampled_colors = all_colors

    train_color_paths, test_color_paths = [], []
    for color_path in tqdm.tqdm(all_colors):
        scene_id = color_path.split('/')[-3]
        seq_id = color_path.split('/')[-2]

        seq_id = osp.join(scene_id, seq_id)

        if seq_id in train_seqs:
            train_color_paths.append(color_path)

        else:
            test_color_paths.append(color_path)

    return train_color_paths, test_color_paths


def read_pose(pose_path):
    return np.loadtxt(pose_path)


def get_rel_trans(ref_pose, src_pose):
    rel_pose = np.linalg.inv(src_pose) @ ref_pose
    rel_trans = np.linalg.norm(rel_pose[:3, -1])

    return rel_trans


def generate_stereo_paths(all_color_paths, interval_range=20, verify_trans=True, lower_thresh=0.05, higher_thresh=0.15, split='train'):
    stereo_pair_paths = []

    for color_path in tqdm.tqdm(all_color_paths):
        src_valid = False
        max_iter = 50
        start_iter = 0

        ref_img_id = color_path.split('/')[-1].split('-')[1].split('.')[0]

        while not src_valid:
            start_iter += 1
            interval = np.random.choice(np.arange(-interval_range, interval_range), size=1, replace=False)[0]

            if interval == 0:
                continue

            src_img_id = str(int(ref_img_id) + interval)
            src_img_id = ('0' * (6 - len(src_img_id))) + src_img_id

            src_path = color_path.replace(ref_img_id, src_img_id)

            if osp.exists(src_path):
                ref_pose_path = color_path.replace('color.png', 'pose.txt')
                src_pose_path = src_path.replace('color.png', 'pose.txt')

                ref_pose = read_pose(ref_pose_path)

                if np.any(np.isnan(ref_pose)) or np.any(np.isinf(ref_pose)) or np.any(np.isneginf(ref_pose)):
                    break

                src_pose = read_pose(src_pose_path)
                if np.any(np.isnan(src_pose)) or np.any(np.isinf(src_pose)) or np.any(np.isneginf(src_pose)):
                    continue

                rel_trans = get_rel_trans(ref_pose, src_pose)

                if rel_trans < lower_thresh or rel_trans > higher_thresh:
                    src_valid = False

                else:
                    src_valid = True
                    break

            if start_iter > max_iter:
                break

        if src_valid:
            stereo_pair_paths.append([color_path, src_path])

    if split == 'train':
        save_path = '7scenes_train_stereo_files.txt'

    elif split == 'test':
        save_path = '7scenes_test_stereo_files.txt'

    with open(save_path, 'w') as fp:
        for pair in stereo_pair_paths:
            ref_path, src_path = pair

            scene = osp.join(ref_path.split('/')[-3], ref_path.split('/')[-2])
            ref_id = ref_path.split('/')[-1]
            src_id = src_path.split('/')[-1]

            fp.write(osp.join(scene, ref_id))
            fp.write('\t')
            fp.write(osp.join(scene, src_id))
            fp.write('\n')


def main(data_root):
    train_color_paths, test_color_paths = get_all_7scenes_path(data_root)
    generate_stereo_paths(train_color_paths, split='train')
    generate_stereo_paths(test_color_paths, split='test')


if __name__ == '__main__':
    # mount your seven-scenes dataset to the local dir "./datasets/seven-scenes"
    data_root = './datasets/seven-scenes'
    main(data_root)
