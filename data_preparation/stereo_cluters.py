import os
import os.path as osp
import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans
from pre_process import load_planes, load_pose, transform_planes


def load_valid_ids(seg_path):
    seg = cv2.imread(seg_path, -1)
    valid_ids = [seg_id for seg_id in np.unique(seg) if seg_id != 65535]
    valid_ids = np.asarray(valid_ids)

    return valid_ids


def kmeans_cluster(vectors, cluster_anchor_num):
    kmeans_N = KMeans(n_clusters=cluster_anchor_num).fit(vectors)
    n_div_d_centers = kmeans_N.cluster_centers_

    return n_div_d_centers


def cluster_vis_3d_scatters(all_n_div_d, cluster_anchor_num=128):
    xs = all_n_div_d[:, 0]
    ys = all_n_div_d[:, 1]
    zs = all_n_div_d[:, 2]

    print('xs:', np.min(xs), np.max(xs), np.mean(xs))
    print('ys:', np.min(ys), np.max(ys), np.mean(ys))
    print('zs:', np.min(zs), np.max(zs), np.mean(zs))

    fig = plt.figure(0)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs, c='b', marker='o')

    anchors = kmeans_cluster(all_n_div_d, cluster_anchor_num=cluster_anchor_num)
    clus_xs = anchors[:, 0]
    clus_ys = anchors[:, 1]
    clus_zs = anchors[:, 2]
    ax.scatter(clus_xs, clus_ys, clus_zs, c='r', marker='^')

    ax.set_label('X Label')
    ax.set_label('Y Label')
    ax.set_label('Z label')

    plt.savefig('n_div_d.png')

    xs = xs.clip(min=-5.5, max=5.5)
    ys = ys.clip(min=-5.5, max=5.5)
    zs = zs.clip(min=-5.5, max=5.5)

    plt.figure(1)
    plt.hist(xs, bins=50, range=(-5, 5))
    plt.savefig('xs.png')

    plt.figure(2)
    plt.hist(ys, bins=50, range=(-5, 5))
    plt.savefig('ys.png')

    plt.figure(3)
    plt.hist(zs, bins=50, range=(-5, 5))
    plt.savefig('zs.png')

    plt.figure(4)
    # plt.xlim(-5, 5)
    sns.displot(xs, color="b", bins=50, kde=True)
    plt.xlim(-5, 5)
    plt.xlabel("X Axis Value")
    plt.ylabel("Density")
    plt.title("Plane Hypothesis X Axis Distribution")
    plt.tight_layout()
    plt.savefig('sns_xs.png')

    plt.figure(5)
    # plt.xlim(-5, 5)
    sns.displot(ys, color="b", bins=50, kde=True)
    plt.xlim(-5, 5)
    plt.xlabel("Y Axis Value")
    plt.ylabel("Density")
    plt.title("Plane Hypothesis Y Axis Distribution")
    plt.tight_layout()
    plt.savefig('sns_ys.png')

    plt.figure(6)
    # plt.xlim(-5, 5)
    sns.displot(zs, color="b", bins=50, kde=True)
    plt.xlim(-5, 5)
    plt.xlabel("Z Axis Value")
    plt.ylabel("Density")
    plt.title("Plane Hypothesis Z Axis Distribution")
    plt.tight_layout()
    plt.savefig('sns_zs.png')

    return anchors


def load_stereo_files(stereo_file):
    paths = []
    with open(stereo_file, 'r') as fp:
        for line in fp:
            ref_path, src_path = line.strip().split('\t')

            paths.append(ref_path)
            paths.append(src_path)

    return paths


def main():
    seg_dir = 'stereo_cleaned_segmentation'
    stereo_file = '../data_splits/valid_stereo_train_files.txt'
    root_data_dir = '/mnt/Data/jiachenliu/scannet_data/scans'
    # data_dir = '/net/SNN-SVM-NAS01/prod_data01/jiachenliu/scannet_data/scans'

    paths = load_stereo_files(stereo_file)
    all_n_div_ds = []

    cluster_anchor_num = 128

    for path in tqdm.tqdm(paths[:10000]):
        scene, img_id = path.split('/')
        seg_path = osp.join(seg_dir, scene, img_id.replace('jpg', 'png'))

        valid_ids = load_valid_ids(seg_path)

        pose_path = osp.join(root_data_dir, scene, 'frames/pose', img_id.replace('jpg', 'txt'))
        pose = load_pose(pose_path)

        plane_path = osp.join(root_data_dir, scene, 'annotation/planes.npy')
        planes = load_planes(plane_path)

        camera_planes = transform_planes(planes, valid_ids, pose)
        n_div_ds = camera_planes[:, :3] / (camera_planes[:, -1:] + 1e-10)

        # avoid nan
        if (n_div_ds != n_div_ds).any():
            continue

        all_n_div_ds.append(n_div_ds)

    all_n_div_ds = np.concatenate(all_n_div_ds, axis=0)
    kmeans_anchors = cluster_vis_3d_scatters(all_n_div_ds, cluster_anchor_num=cluster_anchor_num)

    # np.save('stereo_n_div_d_anchors_{}.npy'.format(cluster_anchor_num), kmeans_anchors)


if __name__ == '__main__':
    main()
