# This file will genearte prompt responses to generate LLM-as-a-judge examples

from datasets import Dataset
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import os
from peft import PeftModel
import json

def get_model_and_tokenizer(model_name_or_path):
    # config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=config)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    return model, tokenizer

def load_model_with_adapter(base_model, adapter_path):
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model

# Prompt taken from https://arxiv.org/pdf/2401.10020.pdf
def format_prompt(ds_row):
    question = ds_row["question"]
    answer = ds_row["answer"]
    prompt = f'''
Review the user’s question and the corresponding response using the additive 5-point
scoring system described below. Points are accumulated based on the satisfaction of each
criterion:
- Add 1 point if the response is relevant and provides some information related to
the user’s inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s question,
but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user’s question in a
useful way, regardless of whether it seems to have been written by an AI Assistant or if it
has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective,
addressing the user’s question directly and comprehensively, and is well-organized and
helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s question
by an AI Assistant, without extraneous information, reflecting expert knowledge, and
demonstrating a high-quality, engaging, and insightful answer.
User: {question}
<response>{answer}</response>
After examining the user’s instruction and the response:
- output the score of the evaluation using this exact format: "score: <total points>", where <total points> is between 0 and 5
- Briefly justify your total score, up to 100 words.
Remember to assess from the AI Assistant perspective, utilizing web search knowledge as
necessary. To evaluate the response in alignment with this additive scoring model, we’ll
systematically attribute points based on the outlined criteria.
        '''
    return prompt

def get_line_count_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        return 0

def model_generate_save(dataset, model, tokenizer, output_file_path):
    ds_lenght = len(dataset)
    start_id = get_line_count_in_file(output_file_path)

    if start_id >= ds_lenght:
        print("No new data to process.")
        return

    dataset = dataset.select(range(start_id, ds_lenght))

    with open(output_file_path, "a") as file:
        for idx, row in enumerate(dataset, start=start_id):
            print(f"Processing: {idx+1}/{ds_lenght}")
            prompt = format_prompt(row)
            rank = row["rank"]
            
            model_inputs = tokenizer(prompt, return_tensors="pt")
            tokens = model_inputs["input_ids"].to("cuda")
            input_length = tokens.shape[1]

            generation_output = model.generate(
                tokens,
                max_new_tokens=115, 
                pad_token_id=tokenizer.eos_token_id
            )
            new_tokens = generation_output[0, input_length:].tolist()  # Get only the new token ids
            output = tokenizer.decode(new_tokens, skip_special_tokens=True)

            result = json.dumps({
                    "prompt": prompt,
                    "response": output,
                    "rank": rank
                })
            file.write(result + '\n')

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_ds_location = os.path.join(script_dir, "datasets/EFT_seed_data/EFT_seed_data_input.jsonl")
    output_file_path = os.path.join(script_dir, "datasets/EFT_seed_data/EFT_seed_data_raw_gen.jsonl")
    input_dataset = Dataset.from_json(input_ds_location)

    model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    model, tokenizer = get_model_and_tokenizer(model_name_or_path)

    # adapter_path = os.path.join(script_dir, '../outputs/Mistral-7B-v0.1-SFT_baseline_IFT')
    # model = load_model_with_adapter(model, adapter_path)
    model.eval()

    model_generate_save(input_dataset, model, tokenizer, output_file_path)

if __name__ == "__main__":
    main()