from datasets import Dataset
import os
import pandas as pd
import re

# This code takes in the generated EFT_seed_data and parses it to find the generated score, EFT_seed_data_1 contains the full list with scores
# EFT_seed_data_2 was filtered to not contain outputs which had no score

script_dir = os.path.dirname(os.path.abspath(__file__))
input_ds_location = os.path.join(script_dir, "./../EFT_seed_data.jsonl")
input_dataset = Dataset.from_json(input_ds_location)

output_data = []

def format_dataset(dataset):
    for idx, row in enumerate(dataset):
        response = row['response']
        prompt = row['prompt']
        # Get score from response substring in format of "Score: x", extract the x
        score_match = re.search(r"Score: (\d+)", response)
        if score_match:
            score = score_match.group(1)  # This captures the first group, which is the score
        else:
            score = -1  # Handle cases where no score is found
        output_data.append((prompt, score, response))
    return pd.DataFrame(output_data, columns=['prompt', 'score', 'response']) 

output_df = format_dataset(input_dataset)
output_df['score'] = pd.to_numeric(output_df['score'], errors='coerce')  # Converts non-convertible types to NaN

score_counts = output_df['score'].value_counts()
print("Score Distribution in Dataset:")
print(score_counts)

output_file_path = os.path.join(script_dir, "EFT_seed_data_1.jsonl")
output_df.to_json(output_file_path, orient="records", lines=True)

# Filter out records that don't have a score from 0 to 5
filtered_df = output_df[(output_df['score'] >= 0) & (output_df['score'] <= 5)]

score_counts = filtered_df['score'].value_counts()
print("Score Distribution in Dataset:")
print(score_counts)

# Output the filtered data to a new JSONL file
filtered_output_file_path = os.path.join(script_dir, "EFT_seed_data_2.jsonl")
filtered_df.to_json(filtered_output_file_path, orient="records", lines=True)