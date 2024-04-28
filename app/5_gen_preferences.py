import pandas as pd
import os
import json

def format_dataset(input_ds_location):
    df = pd.read_json(input_ds_location, lines=True)
    
    # Filter scores and ensure they are in the range 0-5
    df = df[df['score'].between(0, 5)]
    
    # Group by 'prompt_id' and get the entries with the highest and lowest scores
    grouped = df.groupby('prompt_id')
    preference_pairs = []

    for _, group in grouped:

        highest = group.loc[group['score'].idxmax()]
        lowest = group.loc[group['score'].idxmin()]

        if highest['score'] != lowest['score']:  # Ensure they do not have the same score
            preference_pairs.append({
                'prompt': highest['prompt'],
                'prompt_chosen_completion': highest['completion'],
                'prompt_rejected_completion': lowest['completion'],
                'score_rejected': lowest['score'],
                'score_accepted': highest['score']
            })

    # Convert list of dicts to DataFrame
    return pd.DataFrame(preference_pairs)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_ds_location = os.path.join(script_dir, "datasets/generated_scores/generated_scores.jsonl")
    output_file_path = os.path.join(script_dir, "datasets/preference_pairs/preference_pairs.jsonl")
    
    output_df = format_dataset(input_ds_location)
    output_df.to_json(output_file_path, orient="records", lines=True)

if __name__ == "__main__":
    main()
