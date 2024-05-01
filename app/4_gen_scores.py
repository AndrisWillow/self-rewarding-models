import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset
import os
import pandas as pd
import re
import math
import json

# This code generates scoring for all the candidate responses 3 times to account for variance and averages them
# Then outputs a new dataset with the scoring

# This code is made with redundancy, so it's safe to stop the generation and resume it any point
# This was done because the sample generations takes a really long time

# TODO: Add more scalable batching if needed

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

# TODO: move the prompt out to seperate file because it's used in multiple files
def format_prompt(user_question, assitant_answer):
# Prompt taken from https://arxiv.org/pdf/2401.10020.pdf
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
User: {user_question}
<response>{assitant_answer}</response>
After examining the user’s instruction and the response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: “Score: <total points>”
Remember to assess from the AI Assistant perspective, utilizing web search knowledge as
necessary. To evaluate the response in alignment with this additive scoring model, we’ll
systematically attribute points based on the outlined criteria.
        '''
        return prompt

def extract_score_from_text(input_txt):
    score_match = re.search(r"Score: (\d+)", input_txt)
    if score_match:
        score = int(score_match.group(1))  # This captures the first group, which is the score
    else:
        score = -1  # Handle cases where no score is found in generation
    return score

def get_avg_score(output_samples):
    """ Calculates and gets an average score (rounded down) from the list of model outputs """
    total_score, score_found = 0, 0
    for sample in output_samples:
        score = extract_score_from_text(sample)
        if score != -1:
            total_score += score
            score_found += 1
    return math.floor(total_score / score_found) if score_found != 0 else -1

def get_line_count_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        return 0

def model_generate_samples_batched(model, tokenizer, prompt_text, num_samples):
    """Generates multiple samples for the same prompt in a batched manner."""
    prompts = [prompt_text] * num_samples  # Repeat the same prompt to fill the batch
    model_inputs = tokenizer(prompts, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to("cuda")
    output = []

    generation_output = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=115,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_samples  # Ensure we generate the requested number of samples
    )
    responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
    
    for out in responses:
        # Get only the completion part
        completion = out[len(prompt_text):].strip()
        output.append((completion))

    return output

def generate_output(input_ds, model, tokenizer, completion_sample_to_gen, output_file_path):
    with open(output_file_path, "a") as file:
        start_id = get_line_count_in_file(output_file_path) # We have an exact corespondance of line counts in both files
        for idx, _ in enumerate(input_ds, start=start_id):
            print(f"Processing: {idx+1}/{len(input_ds)}")

            prompt_id = input_ds[idx]["prompt_id"]
            prompt = input_ds[idx]["prompt"]
            response = input_ds[idx]["response"]

            full_prompt = format_prompt(prompt, response)
            output_samples = model_generate_samples_batched(model, tokenizer, full_prompt, completion_sample_to_gen)
            avg_score = get_avg_score(output_samples)

            result = json.dumps({
                "prompt_id": prompt_id,
                "prompt": prompt,
                "response": response,
                "avg_score": avg_score,
                "meta_data": output_samples
            })
            file.write(result + '\n')
    
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_ds_location = os.path.join(script_dir, "datasets/generated_responses/generated_responses-2000-3000.jsonl")
    output_file_path = os.path.join(script_dir, "datasets/generated_scores/generated_scores-2000-3000.jsonl")
    adapter_path = os.path.join(script_dir, '../outputs/Mistral-7B-Instruct-v0.2-SFT_baseline_IFT+EFT')

    input_ds = Dataset.from_json(input_ds_location)
    
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
    model, tokenaizer = initialize_model_and_tokenizer(model_name_or_path)

    model = load_model_with_adapter(model, adapter_path)
    model.eval()

    samples_to_gen = 3
    generate_output(input_ds, model, tokenaizer, samples_to_gen, output_file_path)

if __name__ == "__main__":
    main()