# This code will generate new prompts for the model to train on using 8-shot prompting
# to generate new and unique samples based on the IFT seed data.

# TODO: If you have more v-ram resources, adding batched prompting could considerabily speed up the generation process 

from datasets import Dataset
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import os
import pandas as pd
import re
import json

def get_model_and_tokenizer(model_name_or_path):
    """ Initialize and return the model and tokenizer with specific configuration. """
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    model.eval()
    return model, tokenizer

def get_df_from_jsonl(file_path):
    """ Load and return the dataset from the provided JSONL file path. """
    input_dataset = Dataset.from_json(file_path)
    return pd.DataFrame(input_dataset)

def generate_prompt(examples):
    # taken from https://github.com/Oxen-AI/Self-Rewarding-Language-Models/blob/main/scripts/01_gen_prompts.py
    """ Generate a prompt for model input by appending tasks within <task> tags. """
    prompt = """
Come up with a series of tasks and questions. Only the task/question,
no further text/explanation, no additional information.
The task or question should be something a person would ask a chatbot.
"""
    for _, item in enumerate(examples):
        prompt += f"<task>{item}</task>\n"
    return prompt

def get_random_prompts(df, num_selections=8):
    """ Randomly selects and returns a number of prompts from the dataframe. """
    return df.sample(n=num_selections)['question'].tolist()

# Small LLM's have a tendency to get stuck and repeat themsleves, we don't want to have too many of the same prompts in our DS
def filter_new_tasks(new_tasks, seen_prompts):
    """
    Filters new tasks to ensure uniqueness, avoiding duplicates.
    Returns a list of unique tasks that have not been recorded yet.
    """
    unique_tasks = []
    for task in new_tasks:
        clean_task = task.strip()
        if clean_task and clean_task not in seen_prompts:
            seen_prompts.add(clean_task)
            unique_tasks.append(clean_task)
    return unique_tasks

# Because the prompt generation takes a very long time, after each iteration the prompts are written to the file, so it's safe to exit
def append_prompts_to_file(file_path, tasks):
    """ Appends filtered tasks to a JSONL file. """
    with open(file_path, 'a') as file:
        for task in tasks:
            file.write(json.dumps({'prompt': task}) + '\n')

def generate_and_save_prompts(input_df, model, tokenizer, sample_to_gen, output_file_path):
    """ Generates prompts and saves them to a file continuously to avoid data loss on interruption. """
    task_regex = re.compile(r"<task>(.*?)</task>")  # Regex to extract content within <task> tags
    seen_prompts = set()  # Set to track prompts and avoid duplicates
    sample_count = 0

    while sample_count < sample_to_gen:
        task_prompts = get_random_prompts(input_df, 8)
        prompts = generate_prompt(task_prompts)
        model_inputs = tokenizer(prompts, return_tensors="pt")
        tokens = model_inputs["input_ids"].to("cuda")
        input_length = tokens.shape[1] # taking not of input length to only output the model generation

        generation_output = model.generate(
            tokens,
            do_sample=True, 
            temperature=0.6, # Temperature and top_p taken from self-rewarding paper
            top_p=0.9, # also taken from the paper
            max_new_tokens=256, # Max tokens for output generation
            pad_token_id=tokenizer.eos_token_id
        )
        # Decode only the newly generated tokens
        new_tokens = generation_output[0, input_length:].tolist()
        output = tokenizer.decode(new_tokens, skip_special_tokens=True) 
        
        # Extract and remove duplicate tasks
        new_tasks = task_regex.findall(output)
        unique_tasks = filter_new_tasks(new_tasks, seen_prompts)

        append_prompts_to_file(output_file_path, unique_tasks)
        sample_count += len(unique_tasks)
        print(f"Processing: {sample_count}/{sample_to_gen}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_ds_location = os.path.join(script_dir, "datasets/0_EFT_seed_data_input.jsonl")
    output_file_path = os.path.join(script_dir, "datasets/generated_prompts/generated_prompts.jsonl")

    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
    model, tokenizer = get_model_and_tokenizer(model_name_or_path)
    input_df = get_df_from_jsonl(input_ds_location)

    generate_and_save_prompts(input_df, model, tokenizer, 6000, output_file_path)

if __name__ == "__main__":
    main()