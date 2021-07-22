#!/bin/bash

# Usage: ./train_ngc.sh [run-name] [script-path]

set -e
set -v

ngc batch run \
    --instance "dgx1v.16g.2.norm" \
    --name "test_job" \
    --image "nvcr.io/nvidian/robotics/posecnn-pytorch:latest" \
    --result /result \
    --datasetid "8187:/maskrcnn/data/coco" \
    --datasetid "61895:/maskrcnn/data/tabletop" \
    --datasetid "62095:/maskrcnn/data/shapenet-scene" \
    --workspace maskrcnn:/maskrcnn \
    --commandline "sleep 168h"
