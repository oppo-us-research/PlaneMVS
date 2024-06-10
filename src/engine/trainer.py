"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under License: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import datetime
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
# the path 'third_party/maskrcnn_main' has been added to
# system path via sys.path.append('third_party/maskrcnn_main')
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize, print0 
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference


""" load our own modules """
from src.datasets import make_data_loader
from src.utils.metric_logger import TensorboardLogger
from src.tools.utils import tocuda

from torch.cuda import amp


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    scaler,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    meters_train,
    infos,
    do_inference=False
):
    
    myLog_name = f"{cfg.MODEL.OUR_METHOD_NAME}.trainer"
    logger = logging.getLogger(myLog_name)
    logger.info("Start training")

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    #is_distributed = arguments["distributed"]


    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)

    #global_step  = start_iter
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        # NOTE: here batched data returned as list/tuple from dataloader, where
        # `images`: list/tuple of dict, len(images) = batch_size;
        # `targets`: list/tuple of 'BoxList' object, len(targets) = batch_size;
        
        # for example, you can check the details as:
        # print0 (type(images), ", len images = ", len(images), type(targets), ", len targets = ", len(targets))
        # print0 ("\nchecking input images:")
        # for k, v in images[0].items():
        #     if isinstance(v, torch.Tensor):
        #         print0(f"\t{k}: {v.shape}")
        #     else:
        #         print0 (f"\t{k}: {type(v)}")
        
        # print0 ("\n\nchecking input targets:")
        # for k in targets[0].fields():
        #     v = targets[0].get_field(k)
        #     if isinstance(v, torch.Tensor):
        #         print0(f"\t{k}: {v.shape}")
        #     else:
        #         print0 (f"\t{k}: {type(v)}")
        
        """
        > see the outputs:
        <class 'tuple'> , len images =  6 <class 'tuple'> , len targets =  6
        
        checking input images:
            ori_ref_img: <class 'PIL.Image.Image'>
            ori_src_img: <class 'PIL.Image.Image'>
            ref_path: <class 'str'>
            src_path: <class 'str'>
            homo_grid: torch.Size([512, 3, 3])
            hypos: torch.Size([512, 3])
            ref_img: <class 'maskrcnn_benchmark.structures.image_list.ImageList'>
            src_img: <class 'maskrcnn_benchmark.structures.image_list.ImageList'>
        
        checking input targets:
            masks: <class 'maskrcnn_benchmark.structures.segmentation_mask.SegmentationMask'>
            labels: torch.Size([6])
            depth: torch.Size([480, 640])
            intrinsic_for_stereo: torch.Size([4, 4])
            intrinsic: torch.Size([4, 4])
            ref_pose: torch.Size([4, 4])
            src_pose: torch.Size([4, 4])
            img_camera_grid: torch.Size([3, 480, 640])
            planar_mask: torch.Size([480, 640])
            n_div_d_map: torch.Size([480, 640, 3])
        """
        #sys.exit()
        
        
        optimizer.zero_grad()

        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}")
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        if cfg.MODEL.METHOD == 'single': # monocular input
            for img in images:
                img['ref_img'] = img['ref_img'].to(device)
                #img['ref_img'] = tocuda(img['ref_img'])

        elif cfg.MODEL.METHOD == 'stereo' or cfg.MODEL.METHOD == 'refine': 
            # ref+src stereo input pair;
            for img in images:
                img['ref_img'] = img['ref_img'].to(device)
                img['src_img'] = img['src_img'].to(device)
                
                if cfg.MODEL.METHOD == 'stereo':
                    img['hypos'] = img['hypos'].to(device)
                    img['homo_grid'] = img['homo_grid'].to(device)
        
        
        targets = [target.to(device) for target in targets]

        infos.update(
            {'iteration': iteration,
             #'is_distributed': is_distributed
            }
            )
        loss_dict = model(images, targets, infos)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters_train.update(iteration=iteration, loss=losses_reduced, **loss_dict_reduced)
        
        """ with torch.cuda.amp Scaler """
        scaler.scale(losses).backward()
        
        scaler.unscale_(optimizer)
        if cfg.SOLVER.GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                            max_norm=cfg.SOLVER.GRADIENT_MAX_NORM,
                            norm_type=2.0
                        )
        
        scaler.step(optimizer)
        scheduler.step()
        # Updates the scale for next iteration.
        scaler.update()

        batch_time = time.time() - end
        end = time.time()
        meters_train.update(time=batch_time, data=data_time)

        eta_seconds = meters_train.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % cfg.LOG.LOG_INTERVAL == 0 or iteration == max_iter:
            print0(f"iter= {iteration}, meters={str(meters_train)}")

            logger.info(
                meters_train.delimiter.join(
                    [
                        "eta: {eta}", # estimated time of arrival, or left;
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem (MiB): {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_train),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 ,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            print0('Begin validation...')
            meters_val = MetricLogger(delimiter="  ")

            if do_inference:
                synchronize()
                raise NotImplementedError("Our model currently do not support inference during validation")
                _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                    model,
                    # The method changes the segmentation mask format in a data loader,
                    # so every time a new data loader is created:
                    make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=True),
                    dataset_name="[Validation]",
                    iou_types=iou_types,
                    box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                    device=cfg.MODEL.DEVICE,
                    expected_results=cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=None,
                )

            synchronize()
            model.train()
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    if cfg.MODEL.METHOD == 'single':
                        for image_val in images_val:
                            image_val['ref_img'] = image_val['ref_img'].to(device)

                    elif cfg.MODEL.METHOD == 'stereo' or cfg.MODEL.METHOD == 'refine':
                        for image_val in images_val:
                            image_val['ref_img'] = image_val['ref_img'].to(device)
                            image_val['src_img'] = image_val['src_img'].to(device)

                            if cfg.MODEL.METHOD == 'stereo':
                                image_val['homo_grid'] = image_val['homo_grid'].to(device)
                                image_val['hypos'] = image_val['hypos'].to(device)

                    targets_val = [target.to(device) for target in targets_val]
                    
                    loss_dict = model(images_val, targets_val, infos)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(loss=losses_reduced, **loss_dict_reduced)
            synchronize()
            model.train()

            if isinstance(meters_train, TensorboardLogger):
                meters_train.update_val(iteration, meters_val)

            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem (MiB): {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
