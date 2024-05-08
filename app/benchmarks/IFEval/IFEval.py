# Instruction following evaluation
# source: https://github.com/google-research/google-research/tree/master/instruction_following_eval

# This file will genearte prompt responses that will be then evaluated by IFEval

# TODO: Refactor this, add batching
from datasets import Dataset
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import os
from peft import PeftModel
import pandas as pd
import json

def get_model_and_tokenizer(model_name_or_path):
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    return model, tokenizer

def load_model_with_adapter(base_model, adapter_path):
    return PeftModel.from_pretrained(base_model, adapter_path)

def get_line_count_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        return 0

def gen_data_for_eval(dataset, model, tokenizer, output_file_path):
    output_data = []
    total_entries = len(dataset)
    start_id = get_line_count_in_file(output_file_path)
    dataset = dataset.select(range(start_id, total_entries))
    
    with open(output_file_path, "a") as file:
        for idx, row in enumerate(dataset, start=start_id):
            print(f"Processing: {idx+1}/{total_entries}")

            prompt = row["prompt"]
            model_inputs = tokenizer(prompt, return_tensors="pt")

            tokens = model_inputs["input_ids"].to("cuda")

            # Note the length of the input
            input_length = tokens.shape[1]

            generation_output = model.generate(
                tokens,
                max_new_tokens=700, # Output max length
                pad_token_id=tokenizer.eos_token_id
            )
            new_tokens = generation_output[0, input_length:].tolist()  # Get only the new token ids
            output = tokenizer.decode(new_tokens, skip_special_tokens=True)
            output_data.append((prompt, output))
            
            result = json.dumps({
                "prompt": prompt,
                "response": output,
            })
            file.write(result + '\n')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_path = os.path.join(script_dir, '../../../outputs/Mistral-7B-Instruct-v0.2-SFT_baseline_IFT+EFT')
    output_file_path = os.path.join(script_dir, "Mistral-7B-Instruct-v0.2-SFT_baseline_IFT+EFT.jsonl")
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"

    input_ds_location = os.path.join(script_dir, "input_data.jsonl")
    input_dataset = Dataset.from_json(input_ds_location)

    model, tokenizer = get_model_and_tokenizer(model_name_or_path)
    model = load_model_with_adapter(model, adapter_path)
    model.eval()

    gen_data_for_eval(input_dataset, model, tokenizer, output_file_path)

if __name__ == "__main__":
    main()