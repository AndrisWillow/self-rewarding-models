from datasets import load_dataset
import pandas as pd
import os

# This code gets the IFT and EFT(input) seed data from OpenAssistant dataset. EFT data gets generated in the 0.5 file
dataset = load_dataset("OpenAssistant/oasst2")
df = pd.DataFrame(dataset['train'])

# Filter for English prompters with no parent_id (first messages)
prompts_en = df[(df['lang'] == 'en') & (df['role'] == 'prompter') & (df['parent_id'].isna())]
assistant_responses = df[(df['role'] == 'assistant') & (df['lang'] == 'en')]

# Merge the datasets on the 'message_id' of English prompter and 'parent_id' of assistant_responses
conversation_pairs = pd.merge(prompts_en, assistant_responses, left_on='message_id', right_on='parent_id')

# TODO: fix up this part of the script, migth be useful for data visulization
# Distribution of samples per rank
rank_counts = conversation_pairs['rank_y'].value_counts()
print("EFT Data rank distribution:")
for rank, count in rank_counts.items():
    print(f"Rank {rank}: {count} responses")


# # Save the datasets to jsonl files
# script_dir = os.path.dirname(os.path.abspath(__file__))
# ift_output_file_path = os.path.join(script_dir, "datasets/IFT_seed_data/IFT_seed_data.jsonl")
# eft_output_file_path = os.path.join(script_dir, "datasets/EFT_seed_data/EFT_seed_data_input.jsonl")