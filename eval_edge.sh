#!/bin/bash
# Geometric edge model evaluation and visualization
# Usage：bash eval_edge.sh

CKPT=./StereoMatching/IGEV-Improve/EStereo/checkpoints/geo-edge-sceneflow-1.5/geo-edge-sceneflow-1.5_best.pth
DATA_ROOT=./StereoMatching/IGEV-Improve/data/sceneflow
OUTPUT_DIR=./StereoMatching/IGEV-Improve/EStereo/eval_edge_vis
NUM_VIS=100
SPLIT=TEST

CUDA_VISIBLE_DEVICES=0 python eval_edge.py \
    --ckpt $CKPT \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --split $SPLIT \
    --num_vis $NUM_VIS \
    --thresh 0.5 \
    --save_metrics 
