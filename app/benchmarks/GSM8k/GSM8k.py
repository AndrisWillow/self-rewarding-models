from datasets import load_dataset
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import os
from peft import PeftModel
import re
import json
from datetime import datetime
import argparse

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
    prompt = f'''Answer the following math question.  
Question: {question} Explain your reasoning step by step, but don't use more than 100 words. At the end output the answer in this format:<result>x</result> - where x is the result.
Answer: Let's think step by step. ''' 
    return prompt
# TODO fix batching - currently adding a larger size for the batch means some prompts get extra padding to match the largest batched element.
# This seems to cause bad generation
def model_generate_output_batched(model, tokenizer, prompts):
    tokenizer.pad_token = tokenizer.eos_token
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    tokens = model_inputs["input_ids"].to(model.device)
    attention_mask = model_inputs["attention_mask"].to(model.device)

    generation_output = model.generate(
        tokens,
        max_new_tokens=160, # Genearting a few tokens, because there can be noise like new lines etc.
        pad_token_id=tokenizer.pad_token_id,
    )

    output = tokenizer.batch_decode(generation_output[:, tokens.shape[1]:], skip_special_tokens=True)
    return output

# Evaluating the benchmark in a loose way by extracting answers, because the model can be noisy and may output additional things not just an answer
def eval_benchmark_save_results(model, tokenizer, dataset, model_name_or_path, adapter, result_name, batch_size=1):
    model.eval() # setting model in eval form
    score = failed_generations = 0
    total_rows_to_evaluate = len(dataset)
    answer_regex = re.compile(r"<result>(.*?)</result>")
    output_format_regex = re.compile(r"####\s+(\d+)")

    for idx in range(0, total_rows_to_evaluate, batch_size):
        batch_end = min(idx + batch_size, total_rows_to_evaluate)

        prompts = [format_prompt(dataset[row]) for row in range(idx, batch_end)]
        answer_keys = [dataset[row]['answer'] for row in range(idx, batch_end)]

        outputs = model_generate_output_batched(model, tokenizer, prompts)
        for output, answer_key in zip(outputs, answer_keys):

            answer_key_match = output_format_regex.search(answer_key)
            output_match = answer_regex.search(output)
            extracted_answer_key = str(answer_key_match.group(1).strip())
            if output_match and answer_key_match:
                extracted_output = str(output_match.group(1).strip())
                extracted_answer_key = str(answer_key_match.group(1).strip())
                # print(f"{extracted_output}=={extracted_answer_key}")
                
                if extracted_output == extracted_answer_key:
                    score += 1
            else:
                # print(prompts)
                # print(output)
                failed_generations += 1

        print(f"Processed {batch_end}/{total_rows_to_evaluate}, current score: {score}, failed generations: {failed_generations}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_file = f"results-{result_name}-{timestamp}.json" 
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
    parser = argparse.ArgumentParser(description="Run model evaluation with optional adapter.")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Model name or path for loading the model")
    parser.add_argument("--result_name", type=str, default="None",
                        help="Adapter name to be appended to model path if not 'None'")
    parser.add_argument("--adapter_name", type=str, default="None",
                        help="Adapter name to be appended to model path if not 'None'")
    args = parser.parse_args()
    model_name_or_path = args.model_name_or_path
    result_name = args.result_name
    adapter_name = args.adapter_name

    model, tokenizer = get_model_and_tokenizer(model_name_or_path)

    if adapter_name != "None":
        adapter_path = os.path.join(script_dir, f'../../../outputs/{adapter_name}')
        model = load_model_with_adapter(model, adapter_path)

    dataset = load_dataset("gsm8k", "main", split="test")
    
    eval_benchmark_save_results(model, tokenizer, dataset, model_name_or_path, adapter_name, result_name, batch_size=1)

if __name__ == "__main__":
    main()