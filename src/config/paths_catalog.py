"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

"""Centralized catalog of paths."""
import os
from os.path import join as pjoin

class DatasetCatalog(object):
    """ dataset root directory """
    #NOTE: change this variable 'DATA_DIR' to other path which works for you, 
    # if you do not want ot organize all the datasets as below: 
    # Here we assume you are in the project root directory, 
    # e.g., "cd ~/planemvs_proj", and then all the datasets (e.g., ScanNet) 
    # used in the experiments have been soft linked into the folder "./datasets/";
    DATA_DIR = "./datasets/"
    
    DATASETS = {
        # this is the main dataloader we used for training on ScanNet;
        "scannet_stereo_train": { 
            #these <key: value> elements should follow the arguments in __init__() of some Dataset;
            # for example: ScannetStereoDataset defined in src/datasets/scannet_stereo.py;
            #you can see the __init__(...) has the same args as below:
            
            "data_root": pjoin(DATA_DIR, "scannet_data/scans"),
            "data_split_file": "splits/sampled_stereo_train_files.txt",
            "cleaned_seg_dir": pjoin(DATA_DIR, "scannet_data/deltas_split_stereo_cleaned_segmentation"),
            "anchor_normal_dir": pjoin(DATA_DIR, "scannet_data/anchor_normals.npy"),
            "semantic_mapping_dir": "scannet_label_mapping",
            "pixel_plane_dir": pjoin(DATA_DIR, "sampled_pixel_planar_map"),
            "use_new_mapping": False,
            "split": "train",
            "mode": "mask"
        },

        # this is the main dataloader we used for validation on ScanNet;
        "scannet_stereo_val": {
            "data_root": pjoin(DATA_DIR, "scannet_data/scans"),
            "data_split_file": "splits/valid_cleaned_stereo_val_files.txt",
            "cleaned_seg_dir": pjoin(DATA_DIR, "scannet_data/deltas_split_stereo_cleaned_segmentation"),
            "anchor_normal_dir": pjoin(DATA_DIR, "scannet_data/anchor_normals.npy"),
            "semantic_mapping_dir": "scannet_label_mapping",
            "pixel_plane_dir": pjoin(DATA_DIR, "scannet_data/sampled_pixel_planar_map"),
            "use_new_mapping": False,
            "split": "val",
            "mode": "mask"
        },
        
        # for code debugging;
        "scannet_stereo_val_sml": {
            "data_root": pjoin(DATA_DIR, "scannet_data/scans"),
            "data_split_file": "splits/valid_cleaned_stereo_val_sml_files.txt",
            "cleaned_seg_dir": pjoin(DATA_DIR, "scannet_data/deltas_split_stereo_cleaned_segmentation"),
            "anchor_normal_dir": pjoin(DATA_DIR, "scannet_data/anchor_normals.npy"),
            "semantic_mapping_dir": "scannet_label_mapping",
            "pixel_plane_dir": pjoin(DATA_DIR, "scannet_data/sampled_pixel_planar_map"),
            "use_new_mapping": False,
            "split": "val",
            "mode": "mask"
        },

        "scannet_train": {
            # these <key: value> elements should follow the arguments in __init__() of some Dataset;
            # for example: ScannetDataset defined in src/datasets/scannet.py;
            # you can see the __init__(...) has the same args as below:
            "data_root": pjoin(DATA_DIR, "scannet_data/scans"),
            "data_split_file": "splits/sampled_train_files.txt",
            "cleaned_seg_dir": pjoin(DATA_DIR, "scannet_data/cleaned_segmentation"),
            "anchor_normal_dir": pjoin(DATA_DIR, "scannet_data/anchor_normals.npy"),
            "semantic_mapping_dir" : "scannet_label_mapping",
            "split": "train",
            "mode": "mask"
        },

        "scannet_val": {
            "data_root": pjoin(DATA_DIR, "scannet_data/scans"),
            "data_split_file": "splits/img_val_files.txt",
            "cleaned_seg_dir": pjoin(DATA_DIR, "scannet_data/cleaned_segmentation"),
            "anchor_normal_dir": pjoin(DATA_DIR, "scannet_data/anchor_normals.npy"),
            "semantic_mapping_dir": "scannet_label_mapping",
            "split": "val",
            "mode": "mask"
        },

        "seven_scenes_stereo_train": {
            "data_root": pjoin(DATA_DIR, "seven-scenes"),
            "data_split_file": "splits/7scenes_train_stereo_files.txt",
            "cleaned_seg_dir": pjoin(DATA_DIR, "seven-scenes/stereo_cleaned_segmentation"),
            "anchor_normal_dir": pjoin(DATA_DIR, "seven-scenes/anchor_normals.npy"),
            "semantic_mapping_dir": "scannet_label_mapping",
            "split": "train",
            "mode": "mask"
        },
        "seven_scenes_stereo_val": {
            "data_root": pjoin(DATA_DIR, "seven-scenes"),
            "data_split_file": "splits/7scenes_test_stereo_files.txt",
            "cleaned_seg_dir": pjoin(DATA_DIR, "seven-scenes/stereo_cleaned_segmentation"),
            "anchor_normal_dir": pjoin(DATA_DIR, "seven-scenes/anchor_normals.npy"),
            "semantic_mapping_dir": "scannet_label_mapping",
            "split": "val",
            "mode": "mask"
        },

        "tum_rgbd_stereo_train": {
            "data_root": pjoin(DATA_DIR, "TUM-RGBD"),
            "data_split_file": "splits/dvmvs_tumrgbd_train_stereo_files.txt",
            "cleaned_seg_dir": pjoin(DATA_DIR, "TUM-RGBD/stereo_cleaned_segmentation"),
            "anchor_normal_dir": pjoin(DATA_DIR, "TUM-RGBD/anchor_normals.npy"),
            "semantic_mapping_dir": "scannet_label_mapping",
            "pose_dir": pjoin(DATA_DIR, "TUM-RGBD/dvmvs_pose_outputs"),
            "split": "train",
            "mode": "mask"
        },

        "tum_rgbd_stereo_val": {
            "data_root": pjoin(DATA_DIR, "TUM-RGBD"),
            "data_split_file": "splits/dvmvs_tumrgbd_test_stereo_files.txt",
            "cleaned_seg_dir": pjoin(DATA_DIR, "TUM-RGBD/stereo_cleaned_segmentation"),
            "anchor_normal_dir": pjoin(DATA_DIR, "TUM-RGBD/anchor_normals.npy"),
            "semantic_mapping_dir": "scannet_label_mapping",
            "pose_dir": pjoin(DATA_DIR, "TUM-RGBD/dvmvs_pose_outputs"),
            "split": "val",
            "mode": "mask"
        },
    }

    @staticmethod
    def get(name):
        if "scannet" in name:
            attrs = DatasetCatalog.DATASETS[name]

            if "stereo" not in name:
                return dict(
                    # factory: save the name of the dataset class, which are included in 
                    # __all__ defined in maskrcnn_benchmark/data/datasets/__init__.py;
                    factory="ScannetDataset", 
                    args=attrs
                )

            else:
                if 'infer' in name:
                    return dict(
                        factory="ScannetStereoInferenceDataset",
                        args=attrs
                    )

                else:
                    return dict(
                        factory="ScannetStereoDataset",
                        args=attrs
                    )

        elif "seven_scenes" in name:
            attrs = DatasetCatalog.DATASETS[name]

            return dict(
                factory="SevenScenesStereoDataset",
                args=attrs
            )

        elif "tum_rgbd" in name:
            attrs = DatasetCatalog.DATASETS[name]

            return dict(
                factory="TUMRGBDStereoDataset",
                args=attrs
            )

        raise RuntimeError("Dataset not available: {}".format(name))
