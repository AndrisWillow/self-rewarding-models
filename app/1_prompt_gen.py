# This code will generate new prompts for the model to train on using 8-shot prompting to generate new and unique samples based on the IFT seed data

from datasets import Dataset
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import os
import pandas as pd

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=config)

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
model.eval()

output_data = []

script_dir = os.path.dirname(os.path.abspath(__file__))
input_ds_location = os.path.join(script_dir, "EFT_seed_data_3.jsonl")
input_dataset = Dataset.from_json(input_ds_location)
input_df = pd.DataFrame(input_dataset)

# Go thorugh dataset
def format_dataset(dataset, model, tokenizer):
    total_entries = len(dataset)
    for idx, row in enumerate(dataset):
        question = row["question"]
        answer = row["answer"]
        model_inputs = tokenizer(prompt, return_tensors="pt")

        tokens = model_inputs["input_ids"].to(device)

        # Note the length of the input
        input_length = tokens.shape[1]

        generation_output = model.generate(
            tokens,
            # do_sample=True, # Picks tokens from the prob. distribution for more creative responses
            # temperature=0.7, # randomness in sampling (higher temp, more creative, but more random, lower, more predictable), effects logits
            # top_p=0.9, # Limits the set of posible next tokens. Does so by cumulatively selecting the most probable tokens from a prob. distribution until it reaches the limit
            # top_k=20,   # Limits the options to the {top_k} most likely options
            max_new_tokens=115, # Output max length
            pad_token_id=tokenizer.eos_token_id
        )
        # Decode only the newly generated tokens, ignoring the input part
        # Subtract input_length from the generated_ids' length to get only new tokens
        new_tokens = generation_output[0, input_length:].tolist()  # Get only the new token ids
        output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # TODO include rank
        output_data.append((prompt, output))
        print(f"Processing: {idx+1}/{total_entries}")
    return pd.DataFrame(output_data, columns=['prompt', 'response']) # Return as prompt, response

output_df = format_dataset(input_dataset, model, tokenizer)
output_file_path = os.path.join(script_dir, "EFT_seed_data.jsonl")
output_df.to_json(output_file_path, orient="records", lines=True)
