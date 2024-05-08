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
    prompt = f'''Answer this multiple choice question. 
Question: {question} Possible answers: {choices_formatted}. Output only the corresponding letter or number to the correct answer. Answer: ''' 
    return prompt

def model_generate_output_batched(model, tokenizer, prompts):
    tokenizer.pad_token = tokenizer.eos_token
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    tokens = model_inputs["input_ids"].to(model.device)
    attention_mask = model_inputs["attention_mask"].to(model.device)

    # print("tokens shape:", tokens.shape)  # Debug print
    # print("attention_mask shape:", attention_mask.shape)  # Debug print

    # print("Tokens before padding:")
    # for idx, token_ids in enumerate(tokens):
    #     print(f"Input {idx + 1} - Input IDs:", token_ids)
    #     print(f"Input {idx + 1} - Attention Mask:", attention_mask[idx])
    #     print("Decoded input:", tokenizer.decode(token_ids, skip_special_tokens=True))
    #     print("\n")

    generation_output = model.generate(
        tokens,
        max_new_tokens=3, # Genearting a few tokens, because there can be noise like new lines etc.
        pad_token_id=tokenizer.pad_token_id,
    )

    # print("Generated tokens:")
    # for token_ids in generation_output:
    #     print(tokenizer.decode(token_ids, skip_special_tokens=True))
    output = tokenizer.batch_decode(generation_output[:, tokens.shape[1]:], skip_special_tokens=True)
    # print("Output")
    # print(output)
    return output

# Evaluating the benchmark in a loose way by extracting answers, because the model can be noisy and may output additional things not just an answer
def eval_benchmark_save_results(model, tokenizer, dataset, model_name_or_path, adapter="None", batch_size=1):
    score = failed_generations = 0
    total_rows_to_evaluate = len(dataset)
    answer_regex = re.compile(r"[ABCD1234]") #posible answers, taken from arc dataset

    for idx in range(0, total_rows_to_evaluate, batch_size):
        batch_end = min(idx + batch_size, total_rows_to_evaluate)

        prompts = [format_prompt(dataset[row]) for row in range(idx, batch_end)]
        answer_keys = [dataset[row]['answerKey'] for row in range(idx, batch_end)]

        outputs = model_generate_output_batched(model, tokenizer, prompts)
        for output, answer_key in zip(outputs, answer_keys):
            match = answer_regex.search(output)
            if match:
                if match.group() == answer_key:
                    score += 1
            else:
                failed_generations += 1

        print(f"Processed batch ending at index {batch_end}, current score: {score}, failed generations: {failed_generations}")

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
    adapter_name = "None" # TODO add shell params
    dataset = load_dataset('ai2_arc', 'ARC-Challenge', split="test")

    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
    model, tokenizer = get_model_and_tokenizer(model_name_or_path)

    # adapter_name = 'Mistral-7B-Instruct-v0.2-SFT_baseline-DPO-M1'
    # adapter_path = os.path.join(script_dir, f'../../../outputs/{adapter_name}')
    # model = load_model_with_adapter(model, adapter_path)
    model.eval()
    eval_benchmark_save_results(model, tokenizer, dataset, model_name_or_path, adapter_name, batch_size=1)

if __name__ == "__main__":
    main()
