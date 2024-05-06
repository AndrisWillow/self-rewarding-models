import datasets
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import json

# Inference takes a long time, you can safely stop and resume this script if necessary

def initialize_model_and_tokenizer(model_name_or_path):
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

def get_line_count_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        return 0

def model_generate_and_save(eval_set, model, tokenizer, output_file_path):
    with open(output_file_path, "a") as file:
        start_id = get_line_count_in_file(output_file_path) # We have an exact corespondance of line counts in both files
        for idx, data in enumerate(eval_set, start=start_id):
            print(f"Processing: {idx+1}/{len(eval_set)}")
            instruction = data["instruction"]
            model_inputs = tokenizer(instruction, return_tensors="pt")
            input_ids = model_inputs["input_ids"].to("cuda")
            input_length = input_ids.shape[1]
            generation_output = model.generate(
                input_ids,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generation_output[0, input_length:].tolist() # get only the output
            output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            result = json.dumps({
                "instruction": instruction,
                "output": output_text,
            })
            file.write(result + '\n')
    
def save_responses_as_json(responses, output_file_path):
    with open(output_file_path, 'w') as f:
        json.dump(responses, f)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_path = os.path.join(script_dir, '../../../outputs/Mistral-7B-Instruct-v0.2-SFT_baseline-DPO-M1')
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"

    # Gen output for reference model
    base_model, tokenizer = initialize_model_and_tokenizer(model_name_or_path)
    base_model.eval()
    output_file_path = os.path.join(script_dir, 'reference_model_outputs.jsonl')
    model_generate_and_save(eval_set, base_model, tokenizer, output_file_path)


    # Gen output for trained model
    trained_model = load_model_with_adapter(base_model, adapter_path)
    trained_model.eval()
    output_file_path = os.path.join(script_dir, 'model_outputs.jsonl')
    model_generate_and_save(eval_set, trained_model, tokenizer, output_file_path)

if __name__ == "__main__":
    main()