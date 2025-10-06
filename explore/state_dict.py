import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
from transformers import AutoModelForCausalLM, LlavaNextForConditionalGeneration
import torch


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
    parser.add_argument("--model_name", type=str, default="Llama3_8B_LLaVA")
    args = parser.parse_args()
    model_name = args.model_name
    print("==============================================")
    print("***** Explore load_state_dict() & state_dict() *****")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==============================================")

    # Load model & tokenizer
    if "LLaVA" not in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_MAP[model_name],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="/work/bbe2549/.cache/huggingface/hub/"
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_MAP[model_name],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="/work/bbe2549/.cache/huggingface/hub/"
        )

    print(type(model.state_dict()))
    for k, v in model.state_dict().items():
        print(k,   ":",   v.shape)

    print(type(model.state_dict()))
    print("Done")



if __name__ == "__main__":
    main()