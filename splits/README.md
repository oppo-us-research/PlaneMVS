# Data Splits

This directory contains the sampled data splits on different datasets for training and validation

## Scannet

`valid_stereo_train_files.txt` and `valid_stereo_val_files.txt` is the stereo pairs we sampled from `data_preparation/stereo_pre_process.py`.

`sampled_stereo_train_files.txt` is the sub-sampled data split (20K stereo image pairs) from `valid_stereo_train_files.txt` to save the training time for experiments. By default we use it to train our stereo model.

`banet_valid_stereo_train_files.txt` and `banet_valid_stereo_val_files.txt` is a larger sampled dataset following the scene splits of [BA-Net](https://openreview.net/pdf?id=B1gabhRcYX). We may use these splits to report numbers in the final published paper.

## 7-Scenes

`7scenes_stereo_files.txt` is the stereo pairs using the same pre-processing steps (from `data_preparation/generate_7scenes_data.py`) sampled from 7-scenes dataset. We only use it for depth inference and evaluation during testing since it does not contain any plane groundtruth.
