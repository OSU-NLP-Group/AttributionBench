#!/bin/bash

export MODEL_DIR=roberta-large-mnli

# 32 1e-5
# ***************** Set parameters here *****************
dataset_version=subset_balanced
model=$(basename $MODEL_DIR)
template=base_c_e
lr=1e-4
num_train_epoches=2
start_gpu_index=0
master_port=11111
per_device_train_batch_size=8
gas=1
nodes=1
# ***************** The followings are auto-calculated parameters *****************
cuda_devices=$(seq -s ',' $start_gpu_index $(($start_gpu_index + $nodes - 1)))
export CUDA_VISIBLE_DEVICES=$cuda_devices
bs=$((gas * nodes))
eval_bs=$((per_device_train_batch_size * 2))
setting=template-${template}-bs${bs}-lr${lr}-gas${gas}
current_time=$(date +"%Y-%m-%d-%H:%M:%S")

echo ${CUDA_VISIBLE_DEVICES}
# make sure you want to do the deletion
# ************************************************************************************
export OUTPUT_DIR=../checkpoints/${model}-${dataset_version}-${setting}

rm -rf $OUTPUT_DIR
# ************************************************************************************

# torchrun --nproc_per_node ${nodes} --master-port ${master_port} ../src/train/run_mixtral_8x7b.py \
# python ../src/train/run_mixtral_8x7b.py \

export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}
torchrun --nproc_per_node ${nodes} --master-port ${master_port} ../src/train/roberta_train.py \
  --model_name_or_path $MODEL_DIR \
  --template ${template} \
  --template_path ../src/prompts.json \
  --dataset_version ${dataset_version} \
  --data_path osunlp/AttributionBench \
  --num_train_samples -1 \
  --output_dir $OUTPUT_DIR \
  --model_max_length 512 \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${eval_bs} \
  --gradient_accumulation_steps ${gas} \
  --num_train_epochs ${num_train_epoches} \
  --evaluation_strategy no \
  --save_strategy epoch \
  --save_total_limit 1 \
  --logging_strategy steps \
  --logging_steps 10 \
  --learning_rate ${lr} \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --report_to wandb