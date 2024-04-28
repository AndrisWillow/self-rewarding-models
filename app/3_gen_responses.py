import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset
import os
import pandas as pd

# TODO: Generate 4 candidate responses for each prompt (~ 3000 * 4 = 12000)

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
    return model, tokenizer

def load_model_with_adapter(base_model, adapter_path):
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model

def generate_and_save_prompts(input_ds, model, tokenizer, comepletion_sample_to_gen):
    """ Generates prompts """
    ds_row_count = len(input_ds)
    output_data = []
    for idx, row in enumerate(input_ds):
        prompt_id = idx+1 # Adding a prompt id to later group by
        question = row["prompt"]
        prompt = f'''User: {question} Assistant: '''
        model_inputs = tokenizer(prompt, return_tensors="pt")
        tokens = model_inputs["input_ids"].to("cuda")
        input_length = tokens.shape[1] # taking not of input length to only output the model generation

        for _ in range(comepletion_sample_to_gen):
            generation_output = model.generate(
                tokens,
                do_sample=True, 
                temperature=0.7, # Temperature and top_p taken from self-rewarding paper
                top_p=0.9, # also taken from the paper
                max_new_tokens=115, # Max tokens for output generation
                pad_token_id=tokenizer.eos_token_id
            )
            # Decode only the newly generated tokens
            new_tokens = generation_output[0, input_length:].tolist()
            output = tokenizer.decode(new_tokens, skip_special_tokens=True) 
            output_data.append((prompt_id, prompt, output))
            # Extract score 
        print(f"Processing: {prompt_id}/{ds_row_count}")
        if idx == 3: break
    return pd.DataFrame(output_data, columns=['prompt_id', 'prompt', 'response'])
    
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_ds_location = os.path.join(script_dir, "datasets/generated_prompts/unique_prompts-1000.jsonl")
    output_file_path = os.path.join(script_dir, "datasets/generated_responses/generated_responses.jsonl")
    adapter_path = 'outputs/Mistral-7B-Instruct-v0.2-SFT_baseline'

    input_ds = Dataset.from_json(input_ds_location)
    
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
    model, tokenaizer = initialize_model_and_tokenizer(model_name_or_path)

    model = load_model_with_adapter(model, adapter_path)
    model.eval() # TODO: What does this specifically do?

    output_df = generate_and_save_prompts(input_ds, model, tokenaizer, 4)
    output_df.to_json(output_file_path, orient="records", lines=True)

if __name__ == "__main__":
    main()