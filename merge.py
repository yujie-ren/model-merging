import argparse
import torch
import os
from transformers import AutoModelForCausalLM, LlavaNextForConditionalGeneration, AutoTokenizer, LlavaNextProcessor
from merge_methods import linear, power_merging
import time
import gc

############################################
# Merging LLM & LLMinLLaVA
# 1) linear (weighted_averaging): LLMinLLaVA_new = merge_coeff*LLM + (1-merge_coeff)*LLMinLLaVA
# 2) power
############################################

MODEL_MAP = {
    'Llama3_8B': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'Llama3_8B_LLaVA': 'llava-hf/llama3-llava-next-8b-hf',  # Llama3_8B_LLaVA
    'Mistral_7B': 'mistralai/Mistral-7B-Instruct-v0.2',
    'Mistral_7B_LLaVA': 'llava-hf/llava-v1.6-mistral-7b-hf',  # Mistral_7B_LLaVA
    'Yi_34B': 'NousResearch/Nous-Hermes-2-Yi-34B',
    'Yi_34B_LLaVA': 'llava-hf/llava-v1.6-34b-hf',  # Yi_34B_LLaVA
    'Vicuna_7B': 'lmsys/vicuna-7b-v1.5',
    'Vicuna_7B_LLaVA': 'llava-hf/llava-v1.6-vicuna-7b-hf',  # Vicuna_7B_LLaVA
    'Vicuna_13B': 'lmsys/vicuna-13b-v1.5',
    'Vicuna_13B_LLaVA': 'llava-hf/llava-v1.6-vicuna-13b-hf',  # Vicuna_13B_LLaVA
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--LLaVA_name", type=str, default="Vicuna_13B_LLaVA")
    parser.add_argument("--merge_method", type=str, default="power_merging", choices=["linear", "power_merging"])  # linear, power_merging
    # For linear merging
    parser.add_argument("--merge_coeff", type=float, default=0.0)  # For linear merging
    # For power merging
    parser.add_argument("--coeff_upper", type=float, default=1.0)  # coefficient upper limit
    parser.add_argument("--coeff_lower", type=float, default=0.0)  # coefficient lower limit
    parser.add_argument("--power", type=float, default=9.0)  # power for power merging
    parser.add_argument("--mode", type=str, default="coeff_increase", choices=["coeff_increase", "coeff_decrease"])
    # Output path
    parser.add_argument("--output_path", type=str, default="../.cache/model_merge/LLaVAnew_op5-2")
    args = parser.parse_args()
    LLaVA_name, merge_method, merge_coeff, coeff_upper, coeff_lower, power, mode, output_path = (
        args.LLaVA_name, args.merge_method, args.merge_coeff,
        args.coeff_upper, args.coeff_lower, args.power, args.mode, args.output_path)

    print("==================================")
    print("***** Merge *****")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==================================")

    #################################################
    # Step 1: Load LLM & LLaVA
    #################################################
    LLM_name = LLaVA_name.split("_LLaVA")[0]

    # Load LLM & tokenizer of LLM
    LLM = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[LLM_name],
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        cache_dir="/work/bbe2549/.cache/huggingface/hub/"
    )
    tokenizer_LLM = AutoTokenizer.from_pretrained(
        MODEL_MAP[LLM_name],
        cache_dir="/work/bbe2549/.cache/huggingface/hub/"
    )
    vocab_LLM = tokenizer_LLM.get_vocab()

    # Load LLaVA & tokenizer of LLaVA
    LLaVA = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_MAP[LLaVA_name],
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        cache_dir="/work/bbe2549/.cache/huggingface/hub/"
    )
    LLMinLLaVA = LLaVA.language_model
    processor_LLaVA = LlavaNextProcessor.from_pretrained(
        MODEL_MAP[LLaVA_name],
        cache_dir="/work/bbe2549/.cache/huggingface/hub/"
    )
    tokenizer_LLaVA = processor_LLaVA.tokenizer
    vocab_LLMinLLaVA = tokenizer_LLaVA.get_vocab()

    #################################################
    # Step 2: Merging LLM & LLMinLLaVA via baselines
    #################################################
    if merge_method == "linear":
        LLMinLLaVA_new = linear.merge(LLM, vocab_LLM, LLMinLLaVA, vocab_LLMinLLaVA, merge_coeff)
    elif merge_method == "power_merging":
        LLMinLLaVA_new = power_merging.merge(LLM, vocab_LLM, LLMinLLaVA, vocab_LLMinLLaVA, coeff_upper, coeff_lower, power, mode)


    #################################################
    # Step 3: Build LLaVA_new & Save it
    #################################################
    # replace LLMinLLaVA
    LLaVA.language_model.load_state_dict(LLMinLLaVA_new.state_dict())

    # Save LLaVA & processor
    if merge_method == "linear":
        save_path = f"{output_path}/{LLaVA_name}-{merge_method}-coeff{merge_coeff}"
    elif merge_method == "power_merging":
        save_path = f"{output_path}/{LLaVA_name}-{merge_method}-upper{coeff_upper}-lower{coeff_lower}-power{power}-{mode}"

    os.makedirs(save_path, exist_ok=True)
    LLaVA.save_pretrained(save_path)
    processor_LLaVA.save_pretrained(save_path)

    print("\nDone!")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {int(end - start)} seconds")
    gc.collect()
    torch.cuda.empty_cache()