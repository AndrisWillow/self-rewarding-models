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
# print(f'conv pairs = {len(conversation_pairs)}')

# Get the IFT dataset from the conversation pairs where rank is 0, best outputs
ift_seed_data = conversation_pairs[(conversation_pairs['rank_y'] == 0)].head(3200)
ift_seed_data['text'] = ift_seed_data.apply(lambda x: f"User: {x['text_x']} Assistant: {x['text_y']}", axis=1)
ift_seed_data = ift_seed_data[['text']]
# print(f'IFT seed data count: {len(ift_seed_data)}')

# Get all the other conversation pairs that are not in the ift_seed_data, so there is no overlap
conversation_pairs_eft = conversation_pairs[~conversation_pairs.index.isin(ift_seed_data.index)]
# print(len(conversation_pairs_eft))

# Select an even distribution of ranked answers that are in the conversation_pairs to get about 3000 examples
rank_counts_needed = 800 # Number of samples per rank, This dataset is skewed to have more samples in the first 0-2 ranks, but we want to try to get as even of a distribution as posible
eft_seed_data = conversation_pairs_eft.groupby('rank_y').apply(lambda x: x.sample(n=min(rank_counts_needed, len(x)), random_state=42)).reset_index(drop=True)
eft_seed_data = eft_seed_data[['text_x', 'text_y', 'rank_y']]
eft_seed_data.columns = ['question', 'answer', 'rank']

# TODO: fix up this part of the script, migth be useful for data visulization
# Distribution of samples per rank
# rank_counts = eft_seed_data['rank_y'].value_counts()
# print("EFT Data rank distribution:")
# for rank, count in rank_counts.items():
#     print(f"Rank {rank}: {count} responses")


# # Save the datasets to jsonl files
script_dir = os.path.dirname(os.path.abspath(__file__))
ift_output_file_path = os.path.join(script_dir, "datasets/IFT_seed_data/IFT_seed_data.jsonl")
eft_output_file_path = os.path.join(script_dir, "datasets/EFT_seed_data/EFT_seed_data_input.jsonl")

ift_seed_data.to_json(ift_output_file_path, orient="records", lines=True)
eft_seed_data.to_json(eft_output_file_path, orient="records", lines=True)
