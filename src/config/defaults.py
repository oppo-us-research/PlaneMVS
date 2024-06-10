"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""
# ------------------------------------------------------------------------------------
# Modified from maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.METHOD = 'stereo'
_C.MODEL.OUR_METHOD_NAME = 'PlaneMVS' # used for method info in the logging;
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.RETINANET_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
_C.MODEL.CLS_AGNOSTIC_BBOX_REG = False

_C.MODEL.ACTIVATE_BN_TRAIN = False

_C.MODEL.IMG_H = 480
_C.MODEL.IMG_W = 640

_C.MODEL.SRCNN = CN()
_C.MODEL.SRCNN.SEPARATE_PRED = True

# ------------------------------
# Added by PlaneMVS's authors;
# ------------------------------
_C.MODEL.STEREO = CN()
_C.MODEL.STEREO.N_HYPOS_PER_AXIS = 8
_C.MODEL.STEREO.AXIS_RANGE = 2.0
_C.MODEL.STEREO.DENSER_Z = False
_C.MODEL.STEREO.FEATURE_TYPE = 'concat'
_C.MODEL.STEREO.FEATURE_REDUCT_CHANNELS = 16
_C.MODEL.STEREO.STEREO_H = 120
_C.MODEL.STEREO.STEREO_W = 160
_C.MODEL.STEREO.USE_ANCHOR_HYPOS = False
_C.MODEL.STEREO.ANCHOR_HYPOS_NUM = 512
_C.MODEL.STEREO.PRETRAIN_DET = False
_C.MODEL.STEREO.APPLY_GN = True
_C.MODEL.STEREO.APPLY_REFINE = True
_C.MODEL.STEREO.POOL_FEATURE = True
_C.MODEL.STEREO.WITH_PIXEL_DEPTH_LOSS = True
_C.MODEL.STEREO.PLANAR_DEPTH_LOSS_WEIGHT = 1.0
_C.MODEL.STEREO.WITH_LOSS_TERM_UNCERTAINTY = True
_C.MODEL.STEREO.MAX_DET_NUM = 10
_C.MODEL.STEREO.LOSS_TERM_NUM = 9
_C.MODEL.STEREO.SOFT_POOL = True
_C.MODEL.STEREO.TRAIN_PIXEL_PLANE = False
_C.MODEL.STEREO.WITH_HARD_MINING = False
_C.MODEL.STEREO.BLOCK_INIT_GRAD = False
_C.MODEL.STEREO.FOCAL_LOSS_REFINE = False
_C.MODEL.STEREO.WITH_DETECTION_CONSISTENCY = False
_C.MODEL.STEREO.WO_SINGLE_DEPTH = True
_C.MODEL.STEREO.REAL_PSEUDO_SAME_SCALE = False
_C.MODEL.STEREO.INFER_SRC_IMG = False
_C.MODEL.STEREO.MERGE_WHEN_TESTING = False
_C.MODEL.STEREO.RAFT_UPSAMPLE = True
_C.MODEL.STEREO.INSTANCE_PLANAR_DEPTH_LOSS = False
_C.MODEL.STEREO.PRED_INSTANCE_PLANAR_DEPTH_LOSS = False
_C.MODEL.STEREO.WITH_PIXEL_GT_PLANE_MAP = False
_C.MODEL.STEREO.USE_ALL_DATA = False
_C.MODEL.STEREO.FINETUNE_WO_PLANE_LOSS = False
_C.MODEL.STEREO.FINETUNE_DEPTH_ON = False
_C.MODEL.STEREO.FINETUNE_DET_LOSS_WEIGHT = 0.1
_C.MODEL.STEREO.DEPTH_STEREO_ON = False
_C.MODEL.STEREO.DEPTH_HYPO_NUM = 256
_C.MODEL.STEREO.MAX_DEPTH_HYPO = 10.0
_C.MODEL.STEREO.WO_DET_LOSS = False
_C.MODEL.STEREO.WITH_PLANE_LOSS = True


_C.MODEL.ROI_MASK_SCORE_HEAD = CN()
_C.MODEL.ROI_MASK_SCORE_HEAD.ACTIVATE = False

_C.MODEL.CONSISTENCY = CN()
_C.MODEL.CONSISTENCY.ACTIVATE = False

# PlaneRCNN refinement settings
_C.MODEL.REFINE = CN()
_C.MODEL.REFINE.REFINE_SIZE = (192, 256)
_C.MODEL.REFINE.DEFAULT_SIZE = (480, 640)
_C.MODEL.REFINE.WITH_MASK_REFINE_LOSS = False
_C.MODEL.REFINE.WITH_WARPING_LOSS = False
_C.MODEL.REFINE.USE_SOFT_MASK = False
_C.MODEL.REFINE.GENERATE_PSEUDO_GT = False
_C.MODEL.REFINE.PSEUDO_LOSS_WEIGHT = 1.0
_C.MODEL.REFINE.GENERATE_OFFLINE_PSEUDO_GT = False
_C.MODEL.REFINE.OFFLINE_PSEUDO_LOSS_WEIGHT = 1.0

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0

# Flips
_C.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.5
_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.CONV_BODY = "R-50-C4"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_RELU = False


# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
_C.MODEL.RPN.ANCHOR_STRIDE = (16,)
# RPN anchor aspect ratios
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
# Apply the post NMS per batch (default) or per image during training
# (default is True to be consistent with Detectron, see Issue #672)
_C.MODEL.RPN.FPN_POST_NMS_PER_BATCH = True
# Custom rpn head, empty to use default conv or separable conv
_C.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.USE_FPN = False
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS = 0.5
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100
_C.MODEL.ROI_HEADS.NMS_PER_CLS = False

_C.MODEL.ROI_HEADS.APPLY_MASK_NMS = False
_C.MODEL.ROI_HEADS.MASK_NMS_THRESH = 0.3


_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
# GN
_C.MODEL.ROI_BOX_HEAD.USE_GN = False
# Dilation
_C.MODEL.ROI_BOX_HEAD.DILATION = 1
_C.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
_C.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4

# Normal head
_C.MODEL.ROI_BOX_HEAD.WO_NORMAL_HEAD = False

_C.MODEL.ANCHOR_NORMAL_NUM = 7

_C.MODEL.DEPTH = CN()
_C.MODEL.DEPTH.LOSS_TYPE = 'L1Loss'


_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Whether or not resize and translate masks to the input image.
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = True
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
# Dilation
_C.MODEL.ROI_MASK_HEAD.DILATION = 1
# GN
_C.MODEL.ROI_MASK_HEAD.USE_GN = False
# use plane dist to compute weight and focal loss
_C.MODEL.ROI_MASK_HEAD.FOCAL_LOSS = False

_C.MODEL.ROI_KEYPOINT_HEAD = CN()
_C.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR = "KeypointRCNNFeatureExtractor"
_C.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR = "KeypointRCNNPredictor"
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_KEYPOINT_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS = tuple(512 for _ in range(8))
_C.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES = 17
_C.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True

# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

_C.MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.RESNETS.WITH_MODULATED_DCN = False
_C.MODEL.RESNETS.DEFORMABLE_GROUPS = 1


# ---------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()

# This is the number of foreground classes and background.
_C.MODEL.RETINANET.NUM_CLASSES = 81

# Anchor aspect ratios to use
_C.MODEL.RETINANET.ANCHOR_SIZES = (32, 64, 128, 256, 512)
_C.MODEL.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_C.MODEL.RETINANET.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_C.MODEL.RETINANET.STRADDLE_THRESH = 0

# Anchor scales per octave
_C.MODEL.RETINANET.OCTAVE = 2.0
_C.MODEL.RETINANET.SCALES_PER_OCTAVE = 3

# Use C5 or P5 to generate P6
_C.MODEL.RETINANET.USE_C5 = True

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET.NUM_CONVS = 4

# Weight for bbox_regression loss
_C.MODEL.RETINANET.BBOX_REG_WEIGHT = 4.0

# Smooth L1 loss beta for bbox regression
_C.MODEL.RETINANET.BBOX_REG_BETA = 0.11

# During inference, #locs to select based on cls score before NMS is performed
# per FPN level
_C.MODEL.RETINANET.PRE_NMS_TOP_N = 1000

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
_C.MODEL.RETINANET.FG_IOU_THRESHOLD = 0.5

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
_C.MODEL.RETINANET.BG_IOU_THRESHOLD = 0.4

# Focal loss parameter: alpha
_C.MODEL.RETINANET.LOSS_ALPHA = 0.25

# Focal loss parameter: gamma
_C.MODEL.RETINANET.LOSS_GAMMA = 2.0

# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
_C.MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, anchors with score > INFERENCE_TH are
# considered for inference
_C.MODEL.RETINANET.INFERENCE_TH = 0.05

# NMS threshold used in RetinaNet
_C.MODEL.RETINANET.NMS_TH = 0.4


# ---------------------------------------------------------------------------- #
# FBNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.FBNET = CN()
_C.MODEL.FBNET.ARCH = "default"
# custom arch
_C.MODEL.FBNET.ARCH_DEF = ""
_C.MODEL.FBNET.BN_TYPE = "bn"
_C.MODEL.FBNET.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
_C.MODEL.FBNET.WIDTH_DIVISOR = 1
_C.MODEL.FBNET.DW_CONV_SKIP_BN = True
_C.MODEL.FBNET.DW_CONV_SKIP_RELU = True

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.DET_HEAD_LAST_SCALE = 1.0
_C.MODEL.FBNET.DET_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.DET_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.KPTS_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.KPTS_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.KPTS_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.MASK_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.MASK_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.MASK_HEAD_STRIDE = 0

# 0 to use all blocks defined in arch_def
_C.MODEL.FBNET.RPN_HEAD_BLOCKS = 0
_C.MODEL.FBNET.RPN_BN_TYPE = ""


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1 # learning rate decay, defualt 0.1;
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 2500
_C.SOLVER.TEST_PERIOD = 0

_C.SOLVER.GRADIENT_CLIP = False
_C.SOLVER.GRADIENT_MAX_NORM = 5

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

_C.SOLVER.TRAIN_BY_EPOCH = False
_C.SOLVER.TUNE_EPOCHS = (7, 9)
_C.SOLVER.MAX_EPOCHS = 10
_C.SOLVER.EPOCH_SAVE_TIMES = 1

_C.LOG = CN()
_C.LOG.LOG_INTERVAL = 100
_C.LOG.LOG_IMAGES = False
# ------------------------------
# Added by PlaneMVS's authors;
# ------------------------------
_C.LOG.LOG_DIR = './logs/tmp'

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 8
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100

# ---------------------------------------------------------------------------- #
# Test-time augmentations for bounding box detection
# See configs/test_time_aug/e2e_mask_rcnn_R-50-FPN_1x.yaml for an example
# ---------------------------------------------------------------------------- #
_C.TEST.BBOX_AUG = CN()

# Enable test-time augmentation for bounding box detection if True
_C.TEST.BBOX_AUG.ENABLED = False

# Horizontal flip at the original scale (id transform)
_C.TEST.BBOX_AUG.H_FLIP = False

# Each scale is the pixel size of an image's shortest side
_C.TEST.BBOX_AUG.SCALES = ()

# Max pixel size of the longer side
_C.TEST.BBOX_AUG.MAX_SIZE = 4000

# Horizontal flip at each scale
_C.TEST.BBOX_AUG.SCALE_H_FLIP = False


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #

# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"

# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False
