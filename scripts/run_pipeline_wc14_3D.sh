#!/bin/bash

# Set parameters
ROOT_DIR="/home/mgutierrez/datasets/WC-2014/"
SPLIT="test"
CFG="config/hrnetv2_w48.yaml"
CFG_L="config/hrnetv2_w48_l.yaml"
WEIGHTS_KP="weights/SV_kp"
WEIGHTS_L="weights/SV_lines"
SAVE_DIR="inference/inference_3D/inference_wc14/"
DEVICE="cuda:0"
KP_TH=0.00561
LINE_TH=0.14599
MAX_REPROJ_ERR=44
GT_FILE="${SAVE_DIR}${SPLIT}_main.zip"
PRED_FILE="${SAVE_DIR}${SPLIT}_main_pred.zip"


# Run inference script
python scripts/inference_sn.py --cfg $CFG --cfg_l $CFG_L --weights_kp $WEIGHTS_KP --weights_line $WEIGHTS_L --root_dir $ROOT_DIR --split $SPLIT --save_dir $SAVE_DIR --kp_th $KP_TH --line_th $LINE_TH --max_reproj_err $MAX_REPROJ_ERR --cuda $DEVICE --main_cam_only

# Run evaluation script
python sn_calibration/src/evalai_camera.py -s $GT_FILE  -p $PRED_FILE 
