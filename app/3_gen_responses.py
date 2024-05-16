import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset
import os
import pandas as pd
import json

# This code generates candidate responses(equal to completion_sample_to_gen) to the unique prompts and then saves them

# This code is made with redundancy, so it's safe to stop the generation and resume it any point
# This was done because the sample generations takes a really long time

# TODO: Add more scalable batching if needed

def initialize_model_and_tokenizer(model_name_or_path):
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
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model

def find_last_prompt_id(file_path):
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            last_line = lines[-1]
            last_prompt_id = json.loads(last_line)['prompt_id']
            return last_prompt_id
    except (FileNotFoundError, IndexError): # File was not present
        return 0
# TODO: split up this function
def generate_and_save_prompts_batched(input_ds, model, tokenizer, completion_sample_to_gen, output_file_path):
    ds_rows_to_process = len(input_ds)
    start_id = find_last_prompt_id(output_file_path)
    if start_id >= ds_rows_to_process:
        print("No new data to process.")
        return

    dataset = input_ds.select(range(start_id, ds_rows_to_process))

    with open(output_file_path, "a") as file:
        for idx, row in enumerate(dataset, start=start_id):
            prompt_id = idx + 1 # Adding a prompt id to later group by
            print(f"Processing: {prompt_id}/{ds_rows_to_process}")
            
            question = row["prompt"]  # TODO: fix out of bounds error, doesn't affect result
            prompt = f"User: {question} Assistant: "
            prompts = [prompt] * completion_sample_to_gen

            model_inputs = tokenizer(prompts, return_tensors="pt")
            tokens = model_inputs["input_ids"].to("cuda")

            generation_output = model.generate(
                tokens,
                do_sample=True,
                temperature=0.7, # Temperature and top_p taken from self-rewarding paper
                top_p=0.9, # also taken from the paper
                max_new_tokens=255,
                pad_token_id=tokenizer.eos_token_id
            )
            completions = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
            for completion in completions:
                result = json.dumps({
                    'prompt_id': prompt_id, 
                    'prompt': prompt, 
                    'response': completion[len(prompt):].strip() # Extracting only the generated completion
                })
                file.write(result + "\n")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_ds_location = os.path.join(script_dir, "datasets/generated_prompts/unique_prompts.jsonl")
    output_file_path = os.path.join(script_dir, "datasets/generated_responses/generated_responses.jsonl")
    adapter_path = os.path.join(script_dir, '../outputs/Mistral-7B-v0.1-SFT_baseline_IFT+EFT')
    input_ds = Dataset.from_json(input_ds_location)

    model_name_or_path = "mistralai/Mistral-7B-v0.1"
    model, tokenizer = initialize_model_and_tokenizer(model_name_or_path)
    model = load_model_with_adapter(model, adapter_path)
    model.eval()  # Sets the model to evaluation mode, affecting layers like dropout

    completion_sample_to_gen = 4
    generate_and_save_prompts_batched(input_ds, model, tokenizer, completion_sample_to_gen, output_file_path)

if __name__ == "__main__":
    main()
