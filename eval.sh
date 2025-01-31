#!/bin/bash

# Set a random port
MASTER_PORT=$((12000 + $RANDOM % 20000))
# Set the number of threads
OMP_NUM_THREADS=2

BASE_JOB_NAME='conf_1_fake_sora_best_eval_2'
PREFIX='/path/to/workspace'
MODEL_PATH='/path/to/model/internvideo2-L14-k400.bin'

# Assume that on a regular host, you can directly specify the number of GPUs, etc.
GPUS=2  # Adjust the number of GPUs according to your actual situation
GPUS_PER_NODE=1
CPUS_PER_TASK=16

# Specify the subdirectory to be evaluated
DATA_PATH="${PREFIX}/fake_sora_image"

# Get the subdirectory name
SUBDIR_NAME=$(basename ${DATA_PATH})

# Generate JOB_NAME
JOB_NAME="${BASE_JOB_NAME}_${SUBDIR_NAME}"
OUTPUT_DIR="$(dirname $0)/${BASE_JOB_NAME}/${JOB_NAME}"
LOG_DIR="./logs/${BASE_JOB_NAME}/${JOB_NAME}"

# Create the output directory and log directory
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Execute the python command
python run_finetuning_combine.py \
    --model internvideo2_cat_large_patch14_224 \
    --data_path ${DATA_PATH} \
    --prefix ${DATA_PATH} \
    --data_set 'SSV2' \
    --split ',' \
    --nb_classes 2 \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --steps_per_print 10 \
    --batch_size 1 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 8 \
    --num_workers 12 \
    --warmup_epochs 5 \
    --tubelet_size 1 \
    --epochs 100 \
    --lr 2e-5 \
    --drop_path 0.1 \
    --head_drop_path 0.1 \
    --fc_drop_rate 0.0 \
    --layer_decay 0.75 \
    --layer_scale_init_value 1e-5 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --test_best \
    --world_size 1 \
    --start_epoch 0 \
    --eval
