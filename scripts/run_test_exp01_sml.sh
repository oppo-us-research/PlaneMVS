#CUDA_VISIBLE_DEVICES=0 \
#python3 tools/stereo_inference.py \

CONF_YAML_FILE=configs/plane_cfgs/planestereo_sml.yaml
EXP_NAME="exp01-planemvs-epo10-bs6-dgx10"
EXP_NAME_SML="exp01-sml-planemvs-epo10-bs6-dgx10"
#----
CKPT_DIR="./checkpoints"
LOAD_CKPT_PATH="${CKPT_DIR}/$EXP_NAME/model_final.pth"

#LOAD_CKPT_PATH="${CKPT_DIR}/saved/cvpr21_official/model_final.pth"
RESULT_DIR="./results"


GPU_ID=${1:-0}
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python3 -m test_net \
    --config_file $CONF_YAML_FILE \
    --ckpt_file $LOAD_CKPT_PATH \
    --visualize \
    --save_dir $RESULT_DIR/${EXP_NAME_SML} \
    --num_test 1000 \
    --eval \
    --vis_pixel_planar \
    --vis_with_gt \
    --vis_depth_error \
    --use_pixel_planar \
    --vis_gt_mask \
    MODEL.STEREO.WO_SINGLE_DEPTH "True" \
    MODEL.STEREO.DENSER_Z "True" \
    MODEL.STEREO.RAFT_UPSAMPLE "True" \
    MODEL.STEREO.LOSS_TERM_NUM 10 \
