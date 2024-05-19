from datasets import load_dataset
import os
from peft import PeftModel
import re
import json
from datetime import datetime
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'common')))
from model_config import get_model_and_tokenizer

def load_model_with_adapter(base_model, adapter_path):
    return PeftModel.from_pretrained(base_model, adapter_path)

def format_prompt(ds_row):
    question = ds_row['question']
    choices_formatted = " ".join([f"{label}: {text}" for label, text in zip(ds_row['choices']['label'], ds_row['choices']['text'])])
    prompt = f'''Answer this multiple choice question. 
Question: {question} Possible answers: {choices_formatted}. Output only the corresponding letter or number to the correct answer. Answer: ''' 
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
        max_new_tokens=3, # Genearting a few tokens, because there can be noise like new lines etc.
        pad_token_id=tokenizer.pad_token_id,
    )

    output = tokenizer.batch_decode(generation_output[:, tokens.shape[1]:], skip_special_tokens=True)
    return output

# Evaluating the benchmark in a loose way by extracting answers, because the model can be noisy and may output additional things not just an answer
def eval_benchmark_save_results(model, tokenizer, dataset, model_name_or_path, adapter, result_name, batch_size=1):
    model.eval() # setting model in eval form
    score = failed_generations = 0
    total_rows_to_evaluate = len(dataset)
    answer_regex = re.compile(r"[ABCD1234]") #posible answers, taken from arc dataset

    for idx in range(0, total_rows_to_evaluate, batch_size):
        batch_end = min(idx + batch_size, total_rows_to_evaluate)

        prompts = [format_prompt(dataset[row]) for row in range(idx, batch_end)]
        answer_keys = [dataset[row]['answerKey'] for row in range(idx, batch_end)]

        outputs = model_generate_output_batched(model, tokenizer, prompts)
        for output, answer_key in zip(outputs, answer_keys):
            # print(output, answer_key)
            match = answer_regex.search(output)
            if match:
                model_answer = str(match.group().strip())
                answer_key = str(answer_key)
                if model_answer == answer_key:
                    score += 1
            else:
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

    dataset = load_dataset('ai2_arc', 'ARC-Easy', split="test")
    
    eval_benchmark_save_results(model, tokenizer, dataset, model_name_or_path, adapter_name, result_name, batch_size=1)

if __name__ == "__main__":
    main()