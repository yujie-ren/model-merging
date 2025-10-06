import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from PIL import Image
import prompt_builder

QUESTION_MAP = {
    'Llama3_8B_LLaVA': 'What percentage of the brain does a human typically use?',
    'Mistral_7B_LLaVA': 'Where did fortune cookies originate?',
    'Yi_34B_LLaVA': 'On what date was the Declaration of Independence officially signed?',
    'Vicuna_7B_LLaVA': 'Who is faster in real life, a tortoise or a hare?',
    'Vicuna_13B_LLaVA': 'According to the Bible, what forbidden fruit did Adam and Eve eat in the Garden of Eden?'
}

# Model Name:
# Llama3_8B_LLaVA
# Mistral_7B_LLaVA
# Vicuna_7B_LLaVA
# Vicuna_13B_LLaVA
# Yi_34B_LLaVA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--LLaVA_name', type=str, default='Llama3_8B_LLaVA')
    parser.add_argument("--merge_method", type=str, default="linear")
    parser.add_argument("--merge_coeff", type=float, default=1.0)
    parser.add_argument("--text_only", type=bool, default=True)
    args = parser.parse_args()
    LLaVA_name, merge_method, merge_coeff, text_only = args.LLaVA_name, args.merge_method, args.merge_coeff, args.text_only
    print("==================================")
    print("***** Uploaded Inference *****")
    for k,v in vars(args).items():
        print(f"{k}: {v}")
    print("==================================")

    # Load model & processor
    model_path = f"Yujie-AI/{LLaVA_name}-{merge_method}-coeff{merge_coeff}"

    LLaVA = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/work/bbe2549/.cache/huggingface/hub/"
    )
    processor = LlavaNextProcessor.from_pretrained(
        model_path,
        cache_dir="/work/bbe2549/.cache/huggingface/hub/"
    )

    # Construct inputs
    if text_only:
        # LLaVA_prompt = [
        #     {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]}
        # ]

        Question = QUESTION_MAP[LLaVA_name]
        LLaVA_prompt = prompt_builder.build_LLaVA_prompt(Question, 'prompt1', add_image='noImage', model_name=LLaVA_name)
        prompt_str = processor.apply_chat_template(LLaVA_prompt, add_generation_prompt=True)
        inputs = processor(text=prompt_str, return_tensors="pt").to(LLaVA.device)
    else:
        LLaVA_prompt = [
            {"role": "user", "content": [{"type": "text", "text": "What is shown in this image?"},{"type": "image"}]}
        ]
        prompt_str = processor.apply_chat_template(LLaVA_prompt, add_generation_prompt=True)
        image = Image.open("../image/pingpang.jpg")
        inputs = processor(text=prompt_str, images=image, return_tensors="pt").to(LLaVA.device)

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