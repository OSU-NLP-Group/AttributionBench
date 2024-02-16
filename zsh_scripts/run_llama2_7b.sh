#!/bin/bash

export MODEL_DIR=meta-llama/Llama-2-7b-hf

dataset_versions=("subset_balanced" "overall_balanced" "not_balanced" "full_data")
templates=("base_c_e" "base_c_e_r" "base_q_c_e" "base_q_c_e_r")

for dataset_version in "${dataset_versions[@]}"; do
  for template in "${templates[@]}"; do
    model=$(basename $MODEL_DIR)
    num_train_epoches=2
    start_gpu_index=0
    master_port=11111
    per_device_train_batch_size=1
    lr=1e-5
    gas=4
    nodes=8
    cuda_devices=$(seq -s ',' $start_gpu_index $(($start_gpu_index + $nodes - 1)))
    export CUDA_VISIBLE_DEVICES=$cuda_devices
    bs=$((gas * nodes))
    eval_bs=$((per_device_train_batch_size * 2))
    setting=template-${template}-bs${bs}-lr${lr}-gas${gas}
    current_time=$(date +"%Y-%m-%d-%H:%M:%S")
    export OUTPUT_DIR=../../checkpoints/${model}-${dataset_version}-${setting}
    rm -rf $OUTPUT_DIR
    export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}
    
    # train
    torchrun --nproc_per_node ${nodes} --master-port ${master_port} ../src/train/llama_train.py \
      --model_name_or_path $MODEL_DIR \
      --template ${template} \
      --template_path ../src/prompts.json \
      --dataset_version ${dataset_version} \
      --data_path AttributionBench \
      --num_train_samples -1 \
      --bf16 True \
      --output_dir $OUTPUT_DIR \
      --model_max_length 2048 \
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
      --fsdp 'full_shard auto_wrap' \
      --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
      --tf32 True \
      --report_to wandb
    
    # generate inference results
    python ../src/inference/run_inference.py \
        --method attrbench \
        --data_path AttributionBench \
        --dataset_version ${dataset_version} \
        --template_path ../src/prompts.json \
        --model_name ${OUTPUT_DIR} \
        --bs 4 \
        --split test_ood test \
        --output_dir ../inference_results/${dataset_version} \
        --max_length 2048 \
        --max_new_tokens 6 \
        --template ${template}
  done
done


# attrscore/alpaca-7b train
#####################################################################################
# 0-shot inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name osunlp/attrscore-alpaca-7b \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}

# 32 1e-5
# ***************** Set parameters here *****************
export MODEL_DIR=osunlp/attrscore-alpaca-7b
model=$(basename $MODEL_DIR)
dataset_version=subset_balanced
template=base_c_e
lr=1e-5
num_train_epoches=2
start_gpu_index=0
per_device_train_batch_size=1
gas=4
nodes=8
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
export OUTPUT_DIR=../../${model}-${dataset_version}-${setting}
rm -rf $OUTPUT_DIR
# ************************************************************************************

export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}

# train
torchrun --nproc_per_node $MLP_WORKER_GPU \
    --master_addr $MLP_WORKER_0_HOST \
    --node_rank $MLP_ROLE_INDEX \
    --master_port $MLP_WORKER_0_PORT \
    --nnodes $MLP_WORKER_NUM ../src/train/llama_train.py \
    --model_name_or_path $MODEL_DIR \
    --data_path AttributionBench \
    --template ${template} \
    --template_path ../src/prompts.json \
    --dataset_version ${dataset_version} \
    --data_path AttributionBench \
    --num_train_samples -1 \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --model_max_length 2048 \
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
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --report_to wandb

# inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name ${OUTPUT_DIR} \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}
#####################################################################################