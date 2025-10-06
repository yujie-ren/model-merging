#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

model_name_list=(
#    "Llama3_8B_LLaVA"
#    "Mistral_7B_LLaVA"
    "Vicuna_7B_LLaVA"
    "Vicuna_13B_LLaVA"
#    "Yi_34B_LLaVA"
)

power_list=(
    "1.0" "3.0" "9.0"
)

for model_name in "${model_name_list[@]}"; do
  for power in "${power_list[@]}"; do
    python3 upload_to_hf.py \
    --LLaVA_name "$model_name" \
    --power "$power"
  done
done