#!/bin/bash

# Usage: ./train_ngc.sh [run-name] [script-path] [num-gpus]

set -e
set -v

NAME=$1
SCRIPT=$2

ngc batch run \
    --instance "dgx1v.16g.4.norm" \
    --name "$NAME" \
    --image "nvcr.io/nvidian/robotics/posecnn-pytorch:latest" \
    --result /result \
    --datasetid "8187:/maskrcnn/data/coco" \
    --datasetid "61895:/maskrcnn/data/tabletop" \
    --datasetid "62095:/maskrcnn/data/shapenet-scene" \
    --workspace maskrcnn:/maskrcnn \
    --commandline "cd /maskrcnn; bash $SCRIPT" \
    --total-runtime 7D \
    --port 6006
