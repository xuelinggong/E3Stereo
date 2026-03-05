#!/bin/bash
# Independent training of the geometric edge branch: Reusing IGEV Feature + EdgeHead to learn geometric edges on SceneFlow.
# Data: ./data/sceneflow (requires running gtedge.py beforehand to generate gtedge).

name=geo-edge-sceneflow-1-Spt-it3
restore_ckpt= 
logdir=./checkpoints/$name
batch_size=16
lr=0.0001
num_steps=50000

export CUDA_VISIBLE_DEVICES=0

python train_edge.py \
    --name $name \
    --logdir $logdir \
    --data_root ./data/sceneflow \
    --batch_size $batch_size \
    --lr $lr \
    --num_steps $num_steps \
    --pos_weight 2.0 \
    --mixed_precision \
    --eval_freq 5000 \
    --eval_samples 500 \
    --ods_dist_frac 0.0075 \
    --save_freq 10000 \
    --refine_iters 3
