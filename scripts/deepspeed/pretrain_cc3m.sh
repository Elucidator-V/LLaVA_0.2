#!/bin/bash

WEIGHT_VERSION=v1.3
PROMPT_VERSION=v1
MODEL_VERSION="7b"

source ~/.bashrc
module load compilers/cuda/11.7 compilers/gcc/9.3.0 cudnn/8.4.0.27_cuda11.x nccl/2.17.1-1_cuda11.7-alone
module load anaconda/2021.11 
conda activate Qii
cd /home/bingxing2/home/scx6385/code/wdk/LLaVA

date 
unzip -q cc3m/images.zip -d /dev/shm/cc3m 
cp cc3m/chat.json /dev/shm/cc3m
date

# Pretraining
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/deepspeed/zero3.json \
    --model_name_or_path /home/bingxing2/home/scx6385/code/wdk/vicuna-$MODEL_VERSION-$WEIGHT_VERSION \
    --version $WEIGHT_VERSION \
    --data_path /dev/shm/cc3m/chat.json \
    --image_folder /dev/shm/cc3m \
    --vision_tower /home/bingxing2/home/scx6385/code/wdk/clip-vit-large-patch14/ \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/deepspeed_llava-$MODEL_VERSION-$WEIGHT_VERSION-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
	--run_name llava_pretrain_cc3m

# Extract projector features
python scripts/extract_mm_projector.py \
  --model_name_or_path ./checkpoints/deepspeed_llava-$MODEL_VERSION-$WEIGHT_VERSION-pretrain \
  --output ./checkpoints/mm_projector/deepspeed_llava-$MODEL_VERSION-$WEIGHT_VERSION-pretrain.bin