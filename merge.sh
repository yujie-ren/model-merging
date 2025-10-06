#!/bin/sh

# export CUDA_VISIBLE_DEVICES=1

model_name_list=(
#    "Llama3_8B_LLaVA"
    "Mistral_7B_LLaVA"
    "Vicuna_7B_LLaVA"
    "Vicuna_13B_LLaVA"
    "Yi_34B_LLaVA"
)

merge_coeff_list=(
    "0.2" "0.4" "0.6" "0.8"
)

for model_name in "${model_name_list[@]}"; do
  for merge_coeff in "${merge_coeff_list[@]}"; do
    python3 merge_via_baselines.py \
    --LLaVA_name "$model_name" \
    --merge_method "linear" \
    --merge_coeff "$merge_coeff" \
    --output_path "../.cache/model_merge/LLaVAnew_baselines"
  done
done