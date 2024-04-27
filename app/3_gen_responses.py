import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
import os
import json
import pandas as pd

# TODO: Generate 4 candidate responses for each prompt (~ 3000 * 4 = 12000)

def load_input_data(file_path):
    """ Load and return a panda DataFrame from the provided JSONL file path. """
    input_dataset = Dataset.from_json(file_path)
    return pd.DataFrame(input_dataset)

def initialize_model_and_tokenizer(model_name_or_path):
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

def generate_and_save_prompts(input_ds, model, tokenizer, sample_to_gen, output_file_path):
    """ Generates prompts and saves them to a file continuously to avoid data loss on interruption. """

    for idx, row in enumerate(input_ds):
        question = row['prompt']
        
        prompt = f'''User: {question} Assistant: '''

        model_inputs = tokenizer(prompt, return_tensors="pt")
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

        append_prompts_to_file(output_file_path, unique_tasks)
        sample_count += len(unique_tasks)
        print(f"Processing: {idx}/{sample_to_gen}")

def main():
    # TODO: add an adapter?

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_ds_location = os.path.join(script_dir, "datasets/generated_prompts/generated_prompts.jsonl")
    output_file_path = os.path.join(script_dir, "datasets/generated_responses/generated_responses.jsonl")

    input_df = load_input_data(input_ds_location)
    
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
    model, tokenaizer = initialize_model_and_tokenizer(model_name_or_path)

    adapter_relative_path = ''
    adapter_path = os.path.abspath(adapter_relative_path)

    generate_and_save_prompts(input_df, model, tokenaizer)

if __name__ == "__main__":
    main()