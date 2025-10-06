import argparse
import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
import prompt_builder

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

QUESTION_MAP = {
    'Llama3_8B_LLaVA': 'What percentage of the brain does a human typically use?',
    'Mistral_7B_LLaVA': 'Where did fortune cookies originate?',
    'Yi_34B_LLaVA': 'On what date was the Declaration of Independence officially signed?',
    'Vicuna_7B_LLaVA': 'Who is faster in real life, a tortoise or a hare?',
    'Vicuna_13B_LLaVA': 'According to the Bible, what forbidden fruit did Adam and Eve eat in the Garden of Eden?'
}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--LLaVA_name", type=str, default="Vicuna_13B_LLaVA")
    parser.add_argument("--merge_method", type=str, default="power_merging", choices=["linear", "power_merging"])  # linear, power_merging
    # For linear merging
    parser.add_argument("--merge_coeff", type=float, default=0.0) # For linear merging
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
    print("***** LLaVAnew text inference *****")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==================================")

    # Load LLaVA from local
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
    LLaVA.eval()

    # Construct inputs
    # LLaVA_prompt = [
    #     {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]}
    # ]
    Question = QUESTION_MAP[LLaVA_name]
    LLaVA_prompt = prompt_builder.build_LLaVA_prompt(Question, 'prompt1', add_image='noImage', model_name=LLaVA_name)
    prompt_str = processor.apply_chat_template(LLaVA_prompt, add_generation_prompt=True)
    inputs = processor(text=prompt_str, return_tensors="pt").to(LLaVA.device)

    # Get outputs
    outputs = LLaVA.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # Skip input ids & Decoding
    outputs_ids = outputs[0][inputs.input_ids.shape[-1]:]
    outputs_text = processor.tokenizer.decode(outputs_ids, skip_special_tokens=True)

    print("----------")
    print(outputs_text)
    print("----------")
    print("Done")


if __name__ == "__main__":
    main()