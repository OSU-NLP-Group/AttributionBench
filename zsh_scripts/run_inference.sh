export CUDA_VISIBLE_DEVICES=0

template=base_c_e  # base_q_c_e, base_q_c_e_r, base_c_e_r
dataset_version=subset_balanced

# put the model dir here
export MODEL_DIR=osunlp/attrscore-alpaca-7b

python ../src/inference/run_inference.py \
--method attrscore \
--data_path osunlp/AttributionBench \
--dataset_version ${dataset_version} \
--template_path ../src/prompts.json \
--model_name ${MODEL_DIR} \
--bs 1 \
--split test_ood test \
--output_dir ../inference_results/${dataset_version} \
--max_length 1024 \
--max_new_tokens 6 \
--template ${template}