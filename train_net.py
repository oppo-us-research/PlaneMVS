# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# Copyright (c) 2024 OPPO. All rights reserved.
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (do not reorder)
from third_party.maskrcnn_main.maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import numpy as np
import torch
import sys

""" DDP related """
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

### In order to correctly call third_party/maskrcnn_main/maskrcnn_benchmark ###
### modules, and do not destory the original import format ###
### inside the maskrcnn_benchmark python files ###
sys.path.append('third_party/maskrcnn_main')

""" load modules from third_party.maskrcnn_main.maskrcnn_benchmark """
from third_party.maskrcnn_main.maskrcnn_benchmark.solver import make_lr_scheduler
from third_party.maskrcnn_main.maskrcnn_benchmark.solver import make_optimizer
from third_party.maskrcnn_main.maskrcnn_benchmark.engine.inference import inference
from third_party.maskrcnn_main.maskrcnn_benchmark.utils.collect_env import collect_env_info
from third_party.maskrcnn_main.maskrcnn_benchmark.utils.imports import import_file
from third_party.maskrcnn_main.maskrcnn_benchmark.utils.logger import setup_logger
from third_party.maskrcnn_main.maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from third_party.maskrcnn_main.maskrcnn_benchmark.utils.metric_logger import MetricLogger
from third_party.maskrcnn_main.maskrcnn_benchmark.utils.comm import (
    synchronize, get_rank, print0, is_main_process)

""" load our own moduels """
from src.config import cfg
from src.models.detector import build_detection_model
from src.utils.checkpoint import DetectronCheckpointer
from src.datasets import make_data_loader
from src.engine.trainer import do_train
from src.utils.metric_logger import TensorboardLogger


from termcolor import colored
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def run_train(cfg, local_rank, distributed, tune_from_scratch, 
            resume, use_tensorboard,
            checkpoint_file_you_specify = None
            ):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)


    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    
    scaler = GradScaler(enabled=use_mixed_precision)

    # multi-gpu distributed parallel training
    if distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            #find_unused_parameters=True
            find_unused_parameters=False
        )
    else:
        # for DDP it is OK to not specify the device;
        # just use data.to("cuda"), 
        # the "cuda" will point to the the current cuda device with GPU_ID
        # automatically awared.

        # But for DP, the BoxList type data cannot support data.cuda() 
        # to automatically different specific device right now;
        # so now we do not support DP training. It will cause errors like 
        # F.covn(x, w), `x`(cuda:0) and `w`(cuda:1) not in the same device;
        num_node = 1
        ngpus_per_node = torch.cuda.device_count()
        num_gpus = ngpus_per_node * num_node
        err_message = "But for DP, the BoxList type data cannot support .cuda() " + \
                        "to different specific device right now;" + \
                        "So now we do not support DP training. It will cause errors like " + \
                        "Runtime Error: F.covn2d(x, w), `x`(cuda:0) and `w`(cuda:1) not in the same device."
        assert num_gpus == 1, err_message
        #model = torch.nn.DataParallel(model)
        model.cuda()

    optimizer = make_optimizer(cfg, model)
    arguments = {}
    arguments["iteration"] = 0
    arguments["distributed"] = distributed

    # make training loader and get data length
    data_loader, data_len = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"]
    )

    # train and scheduler the network by epoch, not by iteration
    if cfg.SOLVER.TRAIN_BY_EPOCH:
        tune_epochs_num = len(cfg.SOLVER.TUNE_EPOCHS)
        assert tune_epochs_num >= 1, f"SOLVER.TUNE_EPOCHS should have >= 1 elements. But got {cfg.SOLVER.TUNE_EPOCHS}"
        steps = [cfg.SOLVER.TUNE_EPOCHS[i] * data_len // cfg.SOLVER.IMS_PER_BATCH 
                        for i in range(tune_epochs_num)]
        cfg.SOLVER.STEPS = tuple(steps)
        cfg.freeze()
    
    print0(f'LR epoch-step = {cfg.SOLVER.TUNE_EPOCHS}, aka milestones iter-step: {cfg.SOLVER.STEPS}')

    iteration_per_epoch = data_len // cfg.SOLVER.IMS_PER_BATCH
    infos = {'iteration_per_epoch': iteration_per_epoch}

    print0('Total training steps:', cfg.SOLVER.MAX_ITER)

    output_dir = cfg.OUTPUT_DIR
    scheduler = make_lr_scheduler(cfg, optimizer)

    # model save/load initialization
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )

    extra_checkpoint_data = checkpointer.load(
            cfg.MODEL.WEIGHT, 
            tune_from_scratch = tune_from_scratch, 
            resume=resume,
            checkpoint_file_you_specify = checkpoint_file_you_specify
        )
    # update iteration and other meta data
    if resume:
        arguments.update(extra_checkpoint_data)

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val, data_len = make_data_loader(cfg, is_train=False, 
                                                     is_distributed=distributed, 
                                                     is_for_period=True)
    else:
        data_loader_val = None

    if cfg.SOLVER.TRAIN_BY_EPOCH:
        checkpoint_period = cfg.SOLVER.MAX_ITER // cfg.SOLVER.MAX_EPOCHS // cfg.SOLVER.EPOCH_SAVE_TIMES

    else:
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    
    print0 (f"save_checkpoint_freq = {checkpoint_period}") 

    if use_tensorboard:
        meters = TensorboardLogger(
            log_dir=os.path.join(cfg.LOG.LOG_DIR, 'tensorboard'),
            delimiter="  "
        )
        print0 (colored("[*** Tensorboard log] ", 'red') + \
                f"will saving tensorboard log to {cfg.LOG.LOG_DIR}")

    else:
        meters = MetricLogger(delimiter="  ")

    do_train(
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
        meters,
        infos
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val, _ = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(
                                            output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch PlaneMVS Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--is_tune_from_scratch", 
        type = str, 
        default='false',
        help= 'whether to train/finetune from a pretrained detection model. '
              'Use pretrained detection model by default.'
    )

    parser.add_argument(
        "--is_resume", type = str, default='false',
        help="whether to resume a training process from a saved checkpoint"
    )

    parser.add_argument(
        "--use_tensorboard", help="Use tensorboardX logger (Requires tensorboardX installed)",
        action="store_true",
    )

    parser.add_argument(
        '--machine_name',
        type = str,
        required=True,
        #default='dgx10',
        help= "machine name flag used in the exp name"
        )
    
    """ DDP related configuration """
    parser.add_argument(
        '--is_multiprocessing_distributed',
        type = str,
        default='false',
        help= "ddp training or not"
        )
    
    parser.add_argument(
        '--dist_url',
        type = str,
        default='env://',
        help= "ddp dist_url"
        )
    
    parser.add_argument(
        '--dist_backend',
        type = str,
        default='nccl',
        help= "distributed backend"
        )
    
    parser.add_argument("--num_node",
        type=int,
        default=1,
        help='number of node to DDP'
        ) 
    
    parser.add_argument("--seed",
        type=int,
        default=1234,
        help='seed for random numbers'
        ) 
    
    parser.add_argument("--load_weights_path",
        type=str,
        default='',
        help="checkpoint file to load our model weights"
        )
    

    args = parser.parse_args()
    
    # change str args to bool args
    args.tune_from_scratch = str(args.is_tune_from_scratch).lower() == 'true'
    args.resume = str(args.is_resume).lower() == 'true'
    args.multiprocessing_distributed = str(args.is_multiprocessing_distributed).lower() == 'true'
    
    """ DDP related configuration """
    if not torch.cuda.is_available():
        print('[***] GPU not found, Only CPU! This will be slow! Exit!')
        sys.exit()
    
    num_node = args.num_node
    if args.multiprocessing_distributed and args.dist_url == "env://":
        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        WORLD_SIZE = num_gpus
        ngpus_per_node = num_gpus // num_node
        # NOTE:
        # `torchrun` provides a superset of the functionality as `torch.distributed.launch` 
        # with the following additional functionalities:
        # - Worker failures are handled gracefully by restarting all workers.
        # - Worker RANK and WORLD_SIZE are assigned automatically.
        # - Number of nodes is allowed to change between minimum and maximum sizes (elasticity).
        
        # If your training script reads local rank from a --local_rank cmd argument. 
        # Change your training script to read from the LOCAL_RANK environment variable 
        # as `local_rank = int(os.environ["LOCAL_RANK"])`;
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    else:
        ngpus_per_node = torch.cuda.device_count()
        num_gpus = ngpus_per_node * num_node
        WORLD_SIZE = num_gpus
        LOCAL_RANK = args.local_rank
    
    args.distributed = (num_gpus > 1) and args.multiprocessing_distributed
    print0(f"[***] num_gpus = {num_gpus}, DISTRIBUTED = {args.distributed}") 

    # before starting DDP    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    
    logoutput_dir = cfg.LOG.LOG_DIR
    if logoutput_dir:
        mkdir(logoutput_dir)
    
    # start DDP here if enabled;
    if args.distributed:
        local_rank = LOCAL_RANK
        local_world_size = WORLD_SIZE
        n = torch.cuda.device_count() // local_world_size
        device_ids = list(range(local_rank * n, (local_rank + 1) * n))
        dist.init_process_group(
            #backend="nccl", init_method="env://"
            backend = args.dist_backend, 
            init_method = args.dist_url,
            #world_size = WORLD_SIZE, 
            #rank = LOCAL_RANK
            )
        torch.cuda.set_device(local_rank)
        
        print(
            f"[{os.getpid()}] rank = {dist.get_rank()}, " +
            f"world_size = {dist.get_world_size()}, device_ids = {device_ids}"
            )
        print0(
            f'Config: number of gpus: {num_gpus}, ' + 
            f'number of nodes: {num_node}, ' +
            f'world_size (i.e., gpus_per_node * node_num): {WORLD_SIZE}'
            )
        synchronize()
    
    else:
        local_rank = args.local_rank
    
    myLog_name = cfg.MODEL.OUR_METHOD_NAME
    logger = setup_logger(
        name = myLog_name, 
        save_dir = output_dir, 
        distributed_rank = get_rank(), 
        filename="log.txt"
        )
    
    if is_main_process():
        logger.info("Using {} GPUs".format(num_gpus))
        logger.info(args)

        logger.info("Collecting env info (might take some time)")
        logger.info("\n" + collect_env_info())

        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))

        output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
        logger.info("Saving config into: {}".format(output_config_path))
        
        # save overloaded model config in the output directory
        save_config(cfg, output_config_path)

    #sys.exit()
    run_train(cfg, local_rank, args.distributed, 
            args.tune_from_scratch, args.resume, 
            args.use_tensorboard, 
            checkpoint_file_you_specify = args.load_weights_path
        )


if __name__ == "__main__":
    """
    To avoid this error:
    File "planemvs_proj/third_party/maskrcnn_main/maskrcnn_benchmark/structures/segmentation_mask.py", line 132, in __init__
        self.masks = masks.to(self.device) # send to self.device (e.g., cuda:0);
    File "/usr/local/lib/python3.9/dist-packages/torch/cuda/__init__.py", line 206, in _lazy_init
        raise RuntimeError(
    RuntimeError: Cannot re-initialize CUDA in forked subprocess. 
          To use CUDA with multiprocessing, you must use the 'spawn' start method;
    """
    mp.set_start_method('spawn') # good solution !!!!

    main()
