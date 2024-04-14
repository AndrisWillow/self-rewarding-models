from datasets import load_dataset
import pandas as pd
import os

# Preparing arc dataset for training
def format_dataset(dataset, include_answer=True):
    formatted_data = []

    for row in dataset:
        question = row['question']
        answerKey = ""
        choices_formatted = " ".join([f"{label}: {text}" for label, text in zip(row['choices']['label'], row['choices']['text'])])
        # if include_answer:
        answerKey = row['answerKey']

        text = (f"Answer this multiple choice question."
                f"Question: {question} Possible answers: {choices_formatted}"
                f"Output only the corresponding letter to the correct answer. Answer: {answerKey}")

        formatted_data.append(text)

    return pd.DataFrame(formatted_data, columns=['text'])

# Load the datasets
train_ds = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split="train")
validation_ds = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split="validation")

# Format the datasets
train_df = format_dataset(train_ds, include_answer=True)
validation_df = format_dataset(validation_ds, include_answer=False)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths for the output files
train_file_path = os.path.join(script_dir, "train_dataset.jsonl")
validation_file_path = os.path.join(script_dir, "validation_dataset.jsonl")

# Save the datasets to jsonl files
train_df.to_json(train_file_path, orient="records", lines=True)
validation_df.to_json(validation_file_path, orient="records", lines=True)

print("Datasets have been formatted and saved as JSON Lines files.")
