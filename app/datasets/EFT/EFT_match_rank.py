from datasets import Dataset
import os
import pandas as pd
import re

# TODO: Delete this, this was done purely because I forgot to include rank into EFT_seed_data and generating outputs takes ~8h
# EFT_seed_data_3 contains both score and rank

script_dir = os.path.dirname(os.path.abspath(__file__))
EFT_ds_location = os.path.join(script_dir, "EFT_seed_data_2.jsonl")
EFT_dataset = Dataset.from_json(EFT_ds_location)

input_ds_location = os.path.join(script_dir, "./../EFT_seed_data_input.jsonl")
input_dataset = Dataset.from_json(input_ds_location)

# Convert input_dataset to DataFrame for easier manipulation
input_df = pd.DataFrame(input_dataset)
EFT_df = pd.DataFrame(EFT_dataset)

output_data = []

def format_dataset():
    total_entries = len(input_dataset)
    for idx, EFT_row in EFT_df.iterrows():
        response = EFT_row['response']
        prompt = EFT_row['prompt']
        score = EFT_row['score']
        rank = None  # Initialize rank as None for cases where no match is found

        # Check each row in input_df to find a match
        for _, input_row in input_df.iterrows():
            question = input_row['question']
            answer = input_row['answer']
            # Check if both question and answer are contained within the prompt
            if question in prompt and answer in prompt:
                rank = input_row['rank']  # Set rank if a matching row is found
                break  # Break the loop once the match is found to avoid unnecessary checks
        print(f"Processing: {idx+1}/{total_entries}")
        output_data.append((prompt, response, rank, score))

    return pd.DataFrame(output_data, columns=['prompt', 'response', 'rank', 'score'])

output_df = format_dataset()
output_df['rank'] = pd.to_numeric(output_df['rank'], errors='coerce')  # Convert rank to numeric, handling non-numeric cases

output_file_path = os.path.join(script_dir, "EFT_seed_data_3.jsonl")
output_df.to_json(output_file_path, orient="records", lines=True)
