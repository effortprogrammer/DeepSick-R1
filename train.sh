#!/usr/bin/env bash
CUDA_DEVICES="0,1,2,3,4,5,6,7"
length=${#CUDA_DEVICES}
n_gpu=$(( ( (length + 1) / 2 ) - 1 ))

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \

accelerate launch \
    --num_processes=$n_gpu \
    main.py \
    --wandb True \