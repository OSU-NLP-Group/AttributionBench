#!/bin/bash

test_subset_name="test_all_subset_balanced"
test_ood_subset_name="test_ood_all_subset_balanced"
model="gpt-3.5-turbo-1106"
# model="gpt-4-1106-preview"
have_question="True"
have_reference="True"
prompt_name="base"
prompt_model_name="GPT"

python prompting_gpt3.5_mt.py \
--test_subset_name "$test_subset_name" \
--test_ood_subset_name "$test_ood_subset_name" \
--model "$model" \
--have_question "$have_question" \
--have_reference "$have_reference" \
--prompt_name "$prompt_name" \
--prompt_model_name "$prompt_model_name"