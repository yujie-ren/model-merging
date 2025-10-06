import argparse
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import gc
from huggingface_hub import login
token = "****"
login(token=token)
import tempfile
from pathlib import Path

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
    print("***** Upload to Huggingface *****")
    for k,v in vars(args).items():
        print(f"{k}: {v}")
    print("==================================")

    ########################################################
    # Load model & processor from local
    ########################################################
    if merge_method == "linear":
        local_path = f"../{output_path}/{LLaVA_name}-{merge_method}-coeff{merge_coeff}/"
    elif merge_method == "power_merging":
        local_path = f"../{output_path}/{LLaVA_name}-{merge_method}-upper{coeff_upper}-lower{coeff_lower}-power{power}-{mode}/"

    processor = LlavaNextProcessor.from_pretrained(local_path)
    LLaVA = LlavaNextForConditionalGeneration.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    ########################################################
    # Upload to Huggingface
    ########################################################
    path = Path("/work/bbe2549/.cache/temp")
    path.mkdir(parents=True, exist_ok=True)

    if merge_method == "linear":
        save_path = f"Yujie-AI/{LLaVA_name}-{merge_method}-coeff{merge_coeff}"
    elif merge_method == "power_merging":
        save_path = f"Yujie-AI/{LLaVA_name}-{merge_method}-upper{coeff_upper}-lower{coeff_lower}-power{power}-{mode}"

    with tempfile.TemporaryDirectory(dir="/work/bbe2549/.cache/temp") as temp_path:
        tempfile.tempdir = temp_path
        LLaVA.push_to_hub(save_path)
        processor.push_to_hub(save_path)


if __name__ == "__main__":
    main()
    gc.collect()  # force garbage collection
    torch.cuda.empty_cache()  # clear GPU cache
