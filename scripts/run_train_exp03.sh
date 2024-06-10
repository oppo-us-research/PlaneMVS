#!/bin/bash

#-------
#NOTE:
# 1) run: chmod +x this_file_name
# 2) run: ./this_file_name
#-------

#GPU_IDS_STR='0,1'
#GPU_IDS_STR='0'
GPU_IDS_STR=$1
# change IFS to comma: the IFS variable is set to a comma, 
# which means that the read command will use the comma 
# as the delimiter when splitting the string into an array;
IFS=','
GPU_IDS=($GPU_IDS_STR)
# get the length of the array
GPU_NUM=${#GPU_IDS[@]}
export CUDA_VISIBLE_DEVICES=$GPU_IDS_STR
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, nproc_per_node=$GPU_NUM"

#exit
MACHINE_NAME="$HOSTNAME"
IS_MULTIPROCESSING_DISTRIBUTED='true'

NUM_EPOCHS=10
BATCH_SIZE=16 #RTX-A6, 2 GPUs
NUM_WORKERS=4

BASE_LEARNING_RATE=0.003 #bs=6

# 0.003 (for bs=6) is already large enough now, 
# so we do not scale it linealy if using a
# larger batch size (e.g., bs=16)
#BASE_LEARNING_RATE=0.003 #bs=16

echo "NUM_WORKERS=$NUM_WORKERS, BATCH_SIZE=$BATCH_SIZE"

LOG_FREQUENCY=50


#LR_SCHEDULER="constant"
#LR_SCHEDULER="piecewise_epoch"
#LR_SCHEDULER="onecyclelr"

#LR_EPOCH_STEP="(7, 9)" # epochs#10

# if [ $LR_SCHEDULER == 'constant' ]; then
#     let DUMMY_STEP1=$NUM_EPOCHS+2
#     let DUMMY_STEP2=$NUM_EPOCHS+5
#     LR_EPOCH_STEP="$DUMMY_STEP1-$DUMMY_STEP2" # actually, disable this LR_EPOCH_STEP;
#     echo "constant learning rate:, reset LR_EPOCH_STEP to an inactive larger number, ${LR_EPOCH_STEP}"
# fi

CONF_YAML_FILE='configs/plane_cfgs/planestereo.yaml'
NETWORK_CLASS_NAME='planemvs'
echo "[**] MACHINE_NAME=$MACHINE_NAME, NUM_WORKERS=$NUM_WORKERS"

# > see: MODEL.STEREO.USE_ALL_DATA "True" below;
EXP_ID="exp03-alldata" 
EXP_NAME="${EXP_ID}-${NETWORK_CLASS_NAME}-epo${NUM_EPOCHS}-bs${BATCH_SIZE}-g${GPU_NUM}-${MACHINE_NAME}"
echo $EXP_NAME
#exit

#### resume training from checkpoint;
RESUME_PATH=""

#----
#CKPT_DIR="/nfs/STG/SemanticDenseMapping/changjiang/mobile-stereo-proj/checkpoints"
CKPT_DIR="./checkpoints_nfs"
LOG_DIR="./logs_nfs"

#LOAD_WEIGHTS_PATH="${CKPT_DIR}/exp08A-SimpleStereo_v1.3-sf-Li-D192-epo20-LR1e-3-onecyc-bs16-h320xw320-rtxa6ks10/ckpt_epoch_009.pth.tar"
LOAD_WEIGHTS_PATH=""

if [ "$RESUME_PATH" == ''  ]; then
    IS_RESUME='false'
else
    IS_RESUME='true'
fi

echo "[***] loading yamls=${CONF_YAML_FILE}"
echo "[***] loading ckpt=${LOAD_WEIGHTS_PATH}"



#python3 -m torch.distributed.launch \
#torchrun --standalone --nnodes=1 --nproc_per_node=4 -m tools.train_net \

#flag='false'
flag='true'
if [ $flag == 'true' ]; then 
    # torch.distributed.launch is going to be deprecated in favor of torchrun.
    #python3 -m torch.distributed.launch -m train_net \
    ## Single-node multi-worker
    torchrun --standalone --nnodes=1 \
        --nproc_per_node=$GPU_NUM \
        -m train_net \
        --config-file ${CONF_YAML_FILE} \
        --is_multiprocessing_distributed=${IS_MULTIPROCESSING_DISTRIBUTED} \
        --use_tensorboard \
        --machine_name=${MACHINE_NAME} \
        --is_resume=${IS_RESUME} \
        LOG.LOG_INTERVAL ${LOG_FREQUENCY} \
        LOG.LOG_DIR "${LOG_DIR}/${EXP_NAME}" \
        SOLVER.IMS_PER_BATCH ${BATCH_SIZE} \
        DATALOADER.NUM_WORKERS ${NUM_WORKERS} \
        SOLVER.TUNE_EPOCHS "(7, 9)" \
        SOLVER.MAX_EPOCHS ${NUM_EPOCHS} \
        SOLVER.TEST_PERIOD 1000 \
        TEST.IMS_PER_BATCH ${BATCH_SIZE} \
        SOLVER.EPOCH_SAVE_TIMES 3 \
        SOLVER.BASE_LR ${BASE_LEARNING_RATE} \
        MODEL.STEREO.PLANAR_DEPTH_LOSS_WEIGHT 0.1 \
        MODEL.STEREO.DENSER_Z "True" \
        MODEL.STEREO.USE_ALL_DATA "True" \
        MODEL.STEREO.RAFT_UPSAMPLE "True" \
        MODEL.STEREO.PRED_INSTANCE_PLANAR_DEPTH_LOSS "True" \
        MODEL.STEREO.LOSS_TERM_NUM 10 \
        OUTPUT_DIR "${CKPT_DIR}/${EXP_NAME}"

fi
