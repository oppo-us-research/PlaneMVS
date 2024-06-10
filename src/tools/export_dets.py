"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import os.path as osp
import numpy as np


def export_detections(result, img_path, save_dir, dataset, score_thresh=0.50):
    scores = result.get_field('scores')
    if len(scores) == 0:
        return

    valid = scores > score_thresh

    if valid.sum() == 0:
        return

    masks = result.get_field('mask').squeeze(dim=1)[valid].cpu().numpy()
    labels = result.get_field('labels')[valid].cpu().numpy()
    scores = scores[valid].cpu().numpy()

    if not osp.exists(osp.join(save_dir, 'masks')):
        os.makedirs(osp.join(save_dir, 'masks'))

    if not osp.exists(osp.join(save_dir, 'labels')):
        os.makedirs(osp.join(save_dir, 'labels'))

    if not osp.exists(osp.join(save_dir, 'scores')):
        os.makedirs(osp.join(save_dir, 'scores'))

    if dataset == '7-scenes':
        scene_id, seq_id = img_path.split('/')[:2]
        img_id = img_path.split('/')[-1].replace('color.png', 'npy')

        save_id = scene_id + '_' + seq_id + '_' + img_id

    elif dataset == 'tumrgbd':
        scene_id = img_path.split('/')[0]
        img_id = img_path.split('/')[-1].replace('png', 'npy')

        save_id = scene_id + '_' + img_id

    else:
        raise NotImplementedError

    np.save(osp.join(save_dir, 'masks', save_id), masks)
    np.save(osp.join(save_dir, 'labels', save_id), labels)
    np.save(osp.join(save_dir, 'scores', save_id), scores)

    return
