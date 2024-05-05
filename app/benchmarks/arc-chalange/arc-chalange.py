from datasets import load_dataset
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import os
from peft import PeftModel
import re
import json
from datetime import datetime

def get_model_and_tokenizer(model_name_or_path):
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
    return PeftModel.from_pretrained(base_model, adapter_path)

def format_prompt(ds_row):
    question = ds_row['question']
    choices_formatted = " ".join([f"{label}: {text}" for label, text in zip(ds_row['choices']['label'], ds_row['choices']['text'])])
#     prompt = f'''Answer this multiple choice question. 
# Question: {question} Possible answers: {choices_formatted}. Output the answer in this format <Answer>x</Answer> where x is is the corresponding letter to the correct answer. 
# <Answer>'''
    prompt = f'''Answer this multiple choice question. 
Question: {question} Possible answers: {choices_formatted}. Output only the corresponding letter to the correct answer. Answer: ''' 
    return prompt

def model_generate_output_batched(model, tokenizer, prompts):
    tokenizer.pad_token = tokenizer.eos_token
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    tokens = model_inputs["input_ids"].to(model.device)
    
    generation_output = model.generate(
        tokens,
        max_new_tokens=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    output = tokenizer.batch_decode(generation_output[:, tokens.shape[1]:], skip_special_tokens=True)
    return output

def eval_benchmark(model, tokenizer, dataset, model_name_or_path, adapter="None", batch_size=10):
    score = 0
    total_rows_to_evaluate = len(dataset)
    failed_generations = 0

    for idx in range(0, total_rows_to_evaluate, batch_size):
        batch_end = min(idx + batch_size, total_rows_to_evaluate) # Avoid overflow
        prompts = [format_prompt(dataset[row]) for row in range(idx, batch_end)]
        answer_keys = [dataset[row]['answerKey'] for row in range(idx, batch_end)]

        outputs = model_generate_output_batched(model, tokenizer, prompts)

        for output, answer_key in zip(outputs, answer_keys):
            print(output, answer_key)
            # answer_regex = re.compile(r"(.*?)</Answer>")
            # match = answer_regex.search(output)
            # print(output, answer_key)
            # if match and match.group(1) == answer_key:
            #     score += 1
            if output == answer_key: 
                score += 1 
            else:
                failed_generations += 1
        print(f"Processed batch ending at index {batch_end}, current score: {score}")

    # Calculating result and writing to JSON
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_file = f"benchmark_results{timestamp}.json" 
    result = score / total_rows_to_evaluate
    result_data = {
        "model": model_name_or_path,
        "adapter": adapter,
        "result": result,
        "failed_generations": failed_generations,
        "total_rows_evaluated": total_rows_to_evaluate,
        "total_correct_guesses": score
    }
    with open(os.path.join(os.path.dirname(__file__), output_file), "w") as f:
        json.dump(result_data, f, indent=4)

    return result

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_path = "None" # TODO add shell params
    adapter_path = os.path.join(script_dir, '../../../outputs/Mistral-7B-Instruct-v0.2-SFT_baseline-DPO-M1')
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
    dataset = load_dataset('ai2_arc', 'ARC-Challenge', split="test")

    model, tokenizer = get_model_and_tokenizer(model_name_or_path)
    model = load_model_with_adapter(model, adapter_path)
    model.eval()

    eval_benchmark(model, tokenizer, dataset, model_name_or_path, adapter_path, batch_size=1)

if __name__ == "__main__":
    main()
