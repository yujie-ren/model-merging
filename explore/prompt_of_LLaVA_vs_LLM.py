import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import fastchat
from fastchat import model
from test_script import prompt_builder

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
    parser.add_argument("--model_name", type=str, default="Vicuna_13B")
    args = parser.parse_args()
    model_name = args.model_name
    print("=================================")
    print("Using LLaVA prompt to prompt LLM")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=================================")

    # Load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[model_name],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/work/bbe2549/.cache/huggingface/hub/"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_MAP[model_name],
        cache_dir="/work/bbe2549/.cache/huggingface/hub/"
    )

    # Construct inputs
    conv = fastchat.model.get_conversation_template(MODEL_MAP[model_name])
    # conv.system_message = "Your task is to provide a biography (bio) of specific people."  # system content
    conv.append_message(conv.roles[0], "Who are you?")  # user content 1
    conv.append_message(conv.roles[1], "You can call me Vicuna, and I was trained by Large Model Systems Organization (LMSYS) researchers as a language model.")  # assistant content 1
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Question = QUESTION_MAP["Vicuna_13B_LLaVA"]
    # prompt = prompt_builder.build_Vicuna_prompt(Question, 'prompt1', model_name="Vicuna_13B")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Get outputs
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Skip input_ids & Decoding
    outputs_ids = outputs[0][inputs.input_ids.shape[-1]:]
    outputs_text = tokenizer.decode(outputs_ids, skip_special_tokens=True)

    print("----------")
    print(outputs_text)
    print("----------")


if __name__ == "__main__":
    main()