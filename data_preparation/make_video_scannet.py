import os
import os.path as osp
import glob
import numpy as np
import tqdm


np.random.seed(428)

data_split_file = '/nfs/SHARE/dataset/scannet_data/ScanNet/Tasks/Benchmark/scannetv2_val.txt'
data_dir = '/nfs/SHARE/dataset/scannet_data/scans'
sample_rate = 5

save_file = 'scannet_video_test_pairs.txt'


with open(data_split_file, 'r') as fp:
    scenes = [line.strip() for line in fp]
    np.random.shuffle(scenes)


with open(save_file, 'w') as fp:
    for scene in tqdm.tqdm(scenes[:10]):
        imgs = glob.glob(osp.join(data_dir, scene, 'frames/color', '*.jpg'))
        sorted_imgs = sorted(imgs, key=lambda x:int(x.split('/')[-1].split('.')[0]))

        sampled_imgs = sorted_imgs[::10][:100]

        for idx, img_path in enumerate(sampled_imgs):
            if idx == len(sampled_imgs) - 1:
                continue

            ref_id = sampled_imgs[idx].split('/')[-1]
            src_id = sampled_imgs[idx + 1].split('/')[-1]

            fp.write(osp.join(scene, ref_id))
            fp.write('\t')
            fp.write(osp.join(scene, src_id))
            fp.write('\n')
