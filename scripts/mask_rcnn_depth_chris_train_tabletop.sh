#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export NGPUS=4

python3 -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py \
  --config-file configs/e2e_mask_rcnn_R_50_FPN_1x_TTOD_Depth_chris.yaml \
