#!/bin/bash

models=("google/flan-ul2" "google/flan-t5-xxl" "google/flan-t5-large" "google/flan-t5-xl")

for model in "${models[@]}"; do
    # 32 1e-5
    # ***************** Set parameters here *****************
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
    export OUTPUT_DIR=../../checkpoints/${model}-${dataset_version}-${setting}
    rm -rf $OUTPUT_DIR
    # ************************************************************************************

    export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}

    # train
    torchrun --nproc_per_node $MLP_WORKER_GPU \
        --master_addr $MLP_WORKER_0_HOST \
        --node_rank $MLP_ROLE_INDEX \
        --master_port $MLP_WORKER_0_PORT \
        --nnodes $MLP_WORKER_NUM ../src/train/flant5_train.py \
        --model_name_or_path $model \
        --data_path osunlp/AttributionBench \
        --template ${template} \
        --template_path ../src/prompts.json \
        --dataset_version ${dataset_version} \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs $num_train_epoches \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size ${eval_bs} \
        --gradient_accumulation_steps ${gas} \
        --evaluation_strategy no \
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
        --data_path osunlp/AttributionBench \
        --dataset_version ${dataset_version} \
        --template_path ../src/prompts.json \
        --model_name ${OUTPUT_DIR} \
        --bs 4 \
        --split test_ood test \
        --output_dir ../inference_results/${dataset_version} \
        --max_length 2048 \
        --max_new_tokens 6 \
        --template ${template}
    
    # zero-shot inference
    export CUDA_VISIBLE_DEVICES="0"
    python ../src/inference/run_inference.py \
        --method attrbench \
        --data_path osunlp/AttributionBench \
        --dataset_version ${dataset_version} \
        --template_path ../src/prompts.json \
        --model_name $model \
        --bs 4 \
        --split test_ood test \
        --output_dir ../inference_results/${dataset_version} \
        --max_length 2048 \
        --max_new_tokens 6 \
        --template ${template}
done
