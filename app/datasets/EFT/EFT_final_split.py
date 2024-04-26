from datasets import Dataset
import os
import pandas as pd
import re

# EFT_seed_data__final is filtered to contain generated data, that matches the human evaluation, by filtering mismatching scores and ranks

script_dir = os.path.dirname(os.path.abspath(__file__))
input_ds_location = os.path.join(script_dir, "EFT_seed_data_3.jsonl")
input_dataset = Dataset.from_json(input_ds_location)
input_df = pd.DataFrame(input_dataset)

# Print score distribution for 'rank'
# score_counts = input_df['rank'].value_counts()
# print("Score Distribution in Dataset:")
# print(score_counts)

# These conditions are very arbitrary, but they should at least eliminate scenarios where very good answers or bad ones are mislabeled by the model
conditions = [
    ((input_df['score'] == 5) & (input_df['rank'].between(0.0, 1.0, inclusive="both"))),
    ((input_df['score'] == 4) & (input_df['rank'].between(0.0, 3.0))),
    ((input_df['score'] == 3) & (input_df['rank'].between(1.0, 4.0))),
    ((input_df['score'] == 2) & (input_df['rank'].between(2.0, 7.0))),
    ((input_df['score'] == 1) & (input_df['rank'].between(3.0, 15.0))),
    ((input_df['score'] == 0) & (input_df['rank'].between(3.0, 15.0)))
]

# Combine conditions to form a final filter
final_filter = pd.concat([input_df.loc[condition] for condition in conditions])

# Print filtered data and check its distribution (optional)
# filtered_score_counts = final_filter['score'].value_counts()
# print("Filtered Score Distribution in Dataset:")
# print(filtered_score_counts)

# Output the filtered DataFrame

# TODO Generate training and evaluation splits

output_file_path = os.path.join(script_dir, "EFT_seed_data_final.jsonl")
final_filter.to_json(output_file_path, orient="records", lines=True)

# Find the inverse directly from the DataFrame
inverse_filter_df = input_df[~input_df.index.isin(final_filter.index)]

# Output the inverse DataFrame
inverse_output_file_path = os.path.join(script_dir, "EFT_seed_data_failures.jsonl")
inverse_filter_df.to_json(inverse_output_file_path, orient="records", lines=True)
