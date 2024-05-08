from datasets import Dataset
import os
import pandas as pd
import re

# This code takes in the generated EFT_seed_data_raw_gen and parses it to find the generated scores
# Then compares that score if it roughly agrees with the human annotators, and then adds it to the finalized DS: EFT_seed_data

def format_dataset(dataset):
    output_data = []
    for _, row in enumerate(dataset):
        response = row['response']
        prompt = row['prompt']
        rank = row['rank']
        score_match = re.search(r"Score: ([0-5])\b", response)
        if score_match:
            score = score_match.group(1)  # This captures the first group, which is the score
        else:
            score = -1  # Handle cases where no score is found
        output_data.append((prompt, score, response, rank))
    return pd.DataFrame(output_data, columns=['prompt', 'score', 'response', 'rank'])

def filter_df_by_conditions(input_df):
    conditions = [
    ((input_df['score'] == 5) & (input_df['rank'].between(0.0, 1.0, inclusive="both"))),
    ((input_df['score'] == 4) & (input_df['rank'].between(0.0, 3.0))),
    ((input_df['score'] == 3) & (input_df['rank'].between(1.0, 4.0))),
    ((input_df['score'] == 2) & (input_df['rank'].between(2.0, 7.0))),
    ((input_df['score'] == 1) & (input_df['rank'].between(3.0, 15.0))),
    ((input_df['score'] == 0) & (input_df['rank'].between(3.0, 15.0)))
    ]

    # Combine conditions to form a final filter
    filtered_df = pd.concat([input_df.loc[condition] for condition in conditions])
    return filtered_df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_ds_location = os.path.join(script_dir, "datasets/EFT_seed_data/EFT_seed_data_raw_gen.jsonl")
    output_ds_location = os.path.join(script_dir, "datasets/EFT_seed_data/EFT_seed_data.jsonl")
    input_dataset = Dataset.from_json(input_ds_location)

    output_df = format_dataset(input_dataset)
    output_df['score'] = pd.to_numeric(output_df['score'], errors='coerce')  # Converts non-convertible types to NaN


    # Output filtered dataset, that sort of coresponds with human annotator ranking
    filtered_df = filter_df_by_conditions(output_df)

    # To see the score distribution
    # score_counts = filtered_df['score'].value_counts()
    # print("Score Distribution in Dataset:")
    # print(score_counts)

    filtered_df.to_json(output_ds_location, orient="records", lines=True)
     
if __name__ == "__main__":
    main()