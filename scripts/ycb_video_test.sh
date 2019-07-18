#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

python tools/test_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x_ycb_video.yaml"
