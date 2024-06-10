import os
import os.path as osp

train_scenes = ['freiburg1_desk', 'freiburg1_room', 'freiburg1_desk2', 'freiburg3_long_office_household']

tum_stereo_file = '../data_splits/dvmvs_tumrgbd_stereo_files.txt'


def read_files(stereo_file):
    with open(stereo_file, 'r') as fp:
        all_data = [line.strip() for line in fp]

    return all_data


def split(all_data_files):
    train_files, val_files = [], []

    for file_name in all_data_files:
        scene = file_name.split('\t')[0].split('/')[0].replace('rgbd_dataset_', '')
        if scene in train_scenes:
            train_files.append(file_name)

        else:
            val_files.append(file_name)

    with open('dvmvs_tumrgbd_train_stereo_files.txt', 'w') as fp:
        for file_name in train_files:
            fp.write(file_name)
            fp.write('\n')

    with open('dvmvs_tumrgbd_test_stereo_files.txt', 'w') as fp:
        for file_name in val_files:
            fp.write(file_name)
            fp.write('\n')


if __name__ == '__main__':
    all_data = read_files(tum_stereo_file)
    split(all_data)
