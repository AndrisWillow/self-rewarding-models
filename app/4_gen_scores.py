import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset
import os
import pandas as pd
import re
import math

# This code generates scoring for all the candidate responses 3 times to account for variance and averages them
# Then outputs a new dataset with the scoring

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

def format_prompt(user_question, assitant_answer):
# TODO: Move the prompt to a sepearte file as it's used in multiple locations
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
    score_match = re.search(r"Score: ([0-5])\b", input_txt)
    if score_match:
        score = int(score_match.group(1))  # This captures the first group, which is the score
    else:
        score = -1  # Handle cases where no score is found in generation
    return score

def model_generate_samples(model, tokenizer, prompt, prompt_input_length, num_samples):
    """  """
    output = []
    for _ in range(num_samples):
        generation_output = model.generate(
            prompt,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=115,
            pad_token_id=tokenizer.eos_token_id
        )
        new_tokens = generation_output[0, prompt_input_length:].tolist()
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        output.append(response)
    return output

def get_avg_score(output_samples):
    """ Calculates and gets an average score (rounded down) from the list of model outputs """
    total_score, score_found = 0, 0
    for sample in output_samples:
        score = extract_score_from_text(sample)
        if score != -1:
            total_score += score
            score_found += 1
    return math.floor(total_score / score_found) if score_found != 0 else -1

def generate_output(input_ds, model, tokenizer, completion_sample_to_gen):
    output_data = []
    for idx, row in enumerate(input_ds):
        prompt = format_prompt(row["prompt"], row["response"])
        model_inputs = tokenizer(prompt, return_tensors="pt")
        tokens = model_inputs["input_ids"].to("cuda")
        prompt_input_length = tokens.shape[1]

        output_samples = model_generate_samples(model, tokenizer, tokens, prompt_input_length, completion_sample_to_gen)
        avg_score = get_avg_score(output_samples)

        output_data.append({
            "prompt_id": row["prompt_id"],
            "prompt": row["prompt"],
            "response": row["response"],
            "avg_score": avg_score,
            "meta_data": output_samples # Including all the model outputs for now for evaluation # TODO: Remove meta_data
        })

        print(f"Processing: {idx+1}/{len(input_ds)}")

    return pd.DataFrame(output_data)
    
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_ds_location = os.path.join(script_dir, "datasets/generated_responses/generated_responses-0-1000.jsonl")
    output_file_path = os.path.join(script_dir, "datasets/generated_scores/generated_scores-0-1000.jsonl")
    adapter_path = 'outputs/Mistral-7B-Instruct-v0.2-SFT_baseline_IFT+EFT'

    input_ds = Dataset.from_json(input_ds_location)
    
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
    model, tokenaizer = initialize_model_and_tokenizer(model_name_or_path)

    model = load_model_with_adapter(model, adapter_path)
    model.eval() # TODO: What does this specifically do?

    samples_to_gen = 3
    output_df = generate_output(input_ds, model, tokenaizer, samples_to_gen)
    output_df.to_json(output_file_path, orient="records", lines=True)

if __name__ == "__main__":
    main()