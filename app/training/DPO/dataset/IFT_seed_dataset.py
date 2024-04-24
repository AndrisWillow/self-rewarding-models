from datasets import load_dataset
import pandas as pd
import os

# Load the dataset
dataset = load_dataset("OpenAssistant/oasst1")
df = pd.DataFrame(dataset['train'])

# Filter for English prompters with no parent_id (first messages)
english_prompter = df[(df['lang'] == 'en') & (df['role'] == 'prompter') & (df['parent_id'].isna())]

# Print the count of rows in the filtered dataset
print(f"Count of English Prompter messages: {len(english_prompter)}")

# Filter english assistant best (rank 0) responses
assistant_responses = df[(df['role'] == 'assistant') & (df['lang'] == 'en') & (df['rank'] == 0)]

duplicates_check = assistant_responses.groupby('parent_id').filter(lambda x: len(x) > 1)

if not duplicates_check.empty:
    print(f"Some questions have multiple rank 0 responses: {duplicates_check['parent_id'].unique()}")

# # Merge the datasets on the 'message_id' of english_prompter and 'parent_id' of assistant_responses
# conversation_pairs = pd.merge(english_prompter, assistant_responses, left_on='message_id', right_on='parent_id')
# print(f"Count of English Prompter messages: {len(conversation_pairs)}")

# # Select and rename columns as necessary to create the final format
# conversation_pairs = conversation_pairs[['text_x', 'text_y']]
# conversation_pairs.columns = ['Question', 'Answer']
# conversation_pairs['text'] = conversation_pairs.apply(lambda x: f"Question: {x['Question']} Answer: {x['Answer']}", axis=1)
# conversation_pairs = conversation_pairs[['text']]

# # Get the directory of the current script
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Define the path for the output file
# output_file_path = os.path.join(script_dir, "IFT_seed_data.jsonl") 

# # Save the dataset to a jsonl file
# conversation_pairs.to_json(output_file_path, orient="records", lines=True)

# # Output the count of merged pairs and confirmation of file save
# print(f"Total conversation pairs: {len(conversation_pairs)}")
# print("Dataset has been formatted and saved as a JSON Lines file.")
