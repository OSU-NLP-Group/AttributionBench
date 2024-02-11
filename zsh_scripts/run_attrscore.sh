#!/bin/bash

export http_proxy=100.66.28.72:3128
export https_proxy=100.66.28.72:3128
export HTTP_PROXY=100.66.28.72:3128
export HTTPS_PROXY=100.66.28.72:3128

# export http_proxy=100.66.27.151:3128
# export https_proxy=100.66.27.151:3128
# export HTTP_PROXY=100.66.27.151:3128
# export HTTPS_PROXY=100.66.27.151:3128

export TMPDIR=/ML-A800/home/xiangyue/yfli/tmp
export HF_HOME=/ML-A800/home/xiangyue/yfli/.cache/huggingface

cd /ML-A800/home/xiangyue/yfli/AttributionBench/zsh_scripts

# export MODEL_DIR=/ML-A800/models/flan-t5-large
# export MODEL_DIR=/ML-A800/home/xiangyue/yfli/hf_models/ul2
export WANDB_ENTITY=flyhero99
export WANDB_PROJECT=attribution-eval-v3.0-newdata

pip install jsonlines
pip install backoff
pip install anthropic


# # inference
# export CUDA_VISIBLE_DEVICES="0"
# python ../src/inference/run_inference.py \
#     --method attrbench \
#     --data_path ../data_1216/AttributionBench \
#     --dataset_version v3.0 \
#     --template_path ../src/prompts.json \
#     --model_name /ML-A800/home/xiangyue/yfli/hf_models/attrscore-llama-7b \
#     --bs 4 \
#     --split test_ood test \
#     --output_dir ../inference_results/v3.0 \
#     --max_length 2048 \
#     --max_new_tokens 6 \
#     --template base_c_e

# t5-xxl-nli-mixture train
#####################################################################################
# 32 1e-5
# ***************** Set parameters here *****************
export MODEL_DIR=/ML-A800/models/t5_xxl_true_nli_mixture
model=$(basename $MODEL_DIR)
dataset_version=v3.0
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
export OUTPUT_DIR=/ML-A100/public/run/research/xiangyue/checkpoints/attribution_models/${model}-${dataset_version}-${setting}
rm -rf $OUTPUT_DIR
# ************************************************************************************

export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}

# train
torchrun --nproc_per_node $MLP_WORKER_GPU \
    --master_addr $MLP_WORKER_0_HOST \
    --node_rank $MLP_ROLE_INDEX \
    --master_port $MLP_WORKER_0_PORT \
    --nnodes $MLP_WORKER_NUM ../src/train/autoais_train.py \
    --model_name_or_path $MODEL_DIR \
    --data_path ../data_1216/AttributionBench \
    --template ${template} \
    --template_path ../src/prompts.json \
    --dataset_version ${dataset_version} \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $num_train_epoches \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size ${eval_bs} \
    --gradient_accumulation_steps ${gas} \
    --evaluation_strategy "no" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 10 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --tf32 True \
    --report_to wandb \
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap 'T5Block'

# inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method autoais \
    --data_path ../data_1216/AttributionBench \
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


# attrscore/flan-t5-xl train
#####################################################################################
# 32 1e-5
# ***************** Set parameters here *****************
export MODEL_DIR=/ML-A800/home/xiangyue/yfli/hf_models/attrscore-flan-t5-xl
model=$(basename $MODEL_DIR)
dataset_version=v3.0
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
export OUTPUT_DIR=/ML-A100/public/run/research/xiangyue/checkpoints/attribution_models/${model}-${dataset_version}-${setting}
rm -rf $OUTPUT_DIR
# ************************************************************************************

export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}

# train
torchrun --nproc_per_node $MLP_WORKER_GPU \
    --master_addr $MLP_WORKER_0_HOST \
    --node_rank $MLP_ROLE_INDEX \
    --master_port $MLP_WORKER_0_PORT \
    --nnodes $MLP_WORKER_NUM ../src/train/flan-T5_train.py \
    --model_name_or_path $MODEL_DIR \
    --data_path ../data_1216/AttributionBench \
    --template ${template} \
    --template_path ../src/prompts.json \
    --dataset_version ${dataset_version} \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $num_train_epoches \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size ${eval_bs} \
    --gradient_accumulation_steps ${gas} \
    --evaluation_strategy "no" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 10 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --tf32 True \
    --report_to wandb \
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap 'T5Block'

# inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path ../data_1216/AttributionBench \
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


# attrscore/llama-7b train
#####################################################################################
# 0-shot inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path ../data_1216/AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name /ML-A800/home/xiangyue/yfli/hf_models/attrscore-llama-7b \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}

# 32 1e-5
# ***************** Set parameters here *****************
export MODEL_DIR=/ML-A800/home/xiangyue/yfli/hf_models/attrscore-llama-7b
model=$(basename $MODEL_DIR)
dataset_version=v3.0
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
export OUTPUT_DIR=/ML-A100/public/run/research/xiangyue/checkpoints/attribution_models/${model}-${dataset_version}-${setting}
rm -rf $OUTPUT_DIR
# ************************************************************************************

export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}

# train
torchrun --nproc_per_node $MLP_WORKER_GPU \
    --master_addr $MLP_WORKER_0_HOST \
    --node_rank $MLP_ROLE_INDEX \
    --master_port $MLP_WORKER_0_PORT \
    --nnodes $MLP_WORKER_NUM ../src/train/run_train_with_question.py \
    --model_name_or_path $MODEL_DIR \
    --data_path ../data_1216/AttributionBench \
    --template ${template} \
    --template_path ../src/prompts.json \
    --dataset_version ${dataset_version} \
    --data_path ../data_1216/AttributionBench \
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
    --data_path ../data_1216/AttributionBench \
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


# attrscore/alpaca-7b train
#####################################################################################
# 0-shot inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path ../data_1216/AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name /ML-A800/models/attrscore-alpaca-7b \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}

# 32 1e-5
# ***************** Set parameters here *****************
export MODEL_DIR=/ML-A800/models/attrscore-alpaca-7b
model=$(basename $MODEL_DIR)
dataset_version=v3.0
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
export OUTPUT_DIR=/ML-A100/public/run/research/xiangyue/checkpoints/attribution_models/${model}-${dataset_version}-${setting}
rm -rf $OUTPUT_DIR
# ************************************************************************************

export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}

# train
torchrun --nproc_per_node $MLP_WORKER_GPU \
    --master_addr $MLP_WORKER_0_HOST \
    --node_rank $MLP_ROLE_INDEX \
    --master_port $MLP_WORKER_0_PORT \
    --nnodes $MLP_WORKER_NUM ../src/train/run_train_with_question.py \
    --model_name_or_path $MODEL_DIR \
    --data_path ../data_1216/AttributionBench \
    --template ${template} \
    --template_path ../src/prompts.json \
    --dataset_version ${dataset_version} \
    --data_path ../data_1216/AttributionBench \
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
    --data_path ../data_1216/AttributionBench \
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


# flan-t5-large train
#####################################################################################
# 0-shot inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path ../data_1216/AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name /ML-A800/models/flan-t5-large \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}

# 32 1e-5
# ***************** Set parameters here *****************
export MODEL_DIR=/ML-A800/models/flan-t5-large
model=$(basename $MODEL_DIR)
dataset_version=v3.0
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
export OUTPUT_DIR=/ML-A100/public/run/research/xiangyue/checkpoints/attribution_models/${model}-${dataset_version}-${setting}
rm -rf $OUTPUT_DIR
# ************************************************************************************

export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}

# train
torchrun --nproc_per_node $MLP_WORKER_GPU \
    --master_addr $MLP_WORKER_0_HOST \
    --node_rank $MLP_ROLE_INDEX \
    --master_port $MLP_WORKER_0_PORT \
    --nnodes $MLP_WORKER_NUM ../src/train/flan-T5_train.py \
    --model_name_or_path $MODEL_DIR \
    --data_path ../data_1216/AttributionBench \
    --template ${template} \
    --template_path ../src/prompts.json \
    --dataset_version ${dataset_version} \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $num_train_epoches \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size ${eval_bs} \
    --gradient_accumulation_steps ${gas} \
    --evaluation_strategy "no" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 10 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --tf32 True \
    --report_to wandb \
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap 'T5Block'

# inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path ../data_1216/AttributionBench \
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

# flan-t5-xl train
#####################################################################################
# 0-shot inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path ../data_1216/AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name /ML-A800/models/flan-t5-xl \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}

# 32 1e-5
# ***************** Set parameters here *****************
export MODEL_DIR=/ML-A800/models/flan-t5-xl
model=$(basename $MODEL_DIR)
dataset_version=v3.0
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
export OUTPUT_DIR=/ML-A100/public/run/research/xiangyue/checkpoints/attribution_models/${model}-${dataset_version}-${setting}
rm -rf $OUTPUT_DIR
# ************************************************************************************

export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}

# train
torchrun --nproc_per_node $MLP_WORKER_GPU \
    --master_addr $MLP_WORKER_0_HOST \
    --node_rank $MLP_ROLE_INDEX \
    --master_port $MLP_WORKER_0_PORT \
    --nnodes $MLP_WORKER_NUM ../src/train/flan-T5_train.py \
    --model_name_or_path $MODEL_DIR \
    --data_path ../data_1216/AttributionBench \
    --template ${template} \
    --template_path ../src/prompts.json \
    --dataset_version ${dataset_version} \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $num_train_epoches \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size ${eval_bs} \
    --gradient_accumulation_steps ${gas} \
    --evaluation_strategy "no" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 10 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --tf32 True \
    --report_to wandb \
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap 'T5Block'

# inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path ../data_1216/AttributionBench \
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

# flan-t5-xxl train
#####################################################################################
# 0-shot inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path ../data_1216/AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name /ML-A800/models/flan-t5-xxl \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}

# 32 1e-5
# ***************** Set parameters here *****************
export MODEL_DIR=/ML-A800/models/flan-t5-xxl
model=$(basename $MODEL_DIR)
dataset_version=v3.0
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
export OUTPUT_DIR=/ML-A100/public/run/research/xiangyue/checkpoints/attribution_models/${model}-${dataset_version}-${setting}
rm -rf $OUTPUT_DIR
# ************************************************************************************

export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}

# train
torchrun --nproc_per_node $MLP_WORKER_GPU \
    --master_addr $MLP_WORKER_0_HOST \
    --node_rank $MLP_ROLE_INDEX \
    --master_port $MLP_WORKER_0_PORT \
    --nnodes $MLP_WORKER_NUM ../src/train/flan-T5_train.py \
    --model_name_or_path $MODEL_DIR \
    --data_path ../data_1216/AttributionBench \
    --template ${template} \
    --template_path ../src/prompts.json \
    --dataset_version ${dataset_version} \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $num_train_epoches \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size ${eval_bs} \
    --gradient_accumulation_steps ${gas} \
    --evaluation_strategy "no" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 10 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --tf32 True \
    --report_to wandb \
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap 'T5Block'

# inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path ../data_1216/AttributionBench \
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


# flan-ul2 train
#####################################################################################
# 0-shot inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path ../data_1216/AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name /ML-A800/models/flan-ul2 \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}

# 32 1e-5
# ***************** Set parameters here *****************
export MODEL_DIR=/ML-A800/models/flan-ul2
model=$(basename $MODEL_DIR)
dataset_version=v3.0
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
export OUTPUT_DIR=/ML-A100/public/run/research/xiangyue/checkpoints/attribution_models/${model}-${dataset_version}-${setting}
rm -rf $OUTPUT_DIR
# ************************************************************************************

export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}

# train
torchrun --nproc_per_node $MLP_WORKER_GPU \
    --master_addr $MLP_WORKER_0_HOST \
    --node_rank $MLP_ROLE_INDEX \
    --master_port $MLP_WORKER_0_PORT \
    --nnodes $MLP_WORKER_NUM ../src/train/flan-T5_train.py \
    --model_name_or_path $MODEL_DIR \
    --data_path ../data_1216/AttributionBench \
    --template ${template} \
    --template_path ../src/prompts.json \
    --dataset_version ${dataset_version} \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $num_train_epoches \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size ${eval_bs} \
    --gradient_accumulation_steps ${gas} \
    --evaluation_strategy "no" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 10 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --tf32 True \
    --report_to wandb \
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap 'T5Block'

# inference
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path ../data_1216/AttributionBench \
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