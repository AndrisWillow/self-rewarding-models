# This file will genearte prompt responses to generate LLM-as-a-judge examples

from datasets import Dataset
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import os
from peft import PeftModel
import pandas as pd

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=config)

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

# With adapter
relative_path = 'outputs/Mistral-7B-Instruct-v0.2-SFT_baseline'
adapter_path = os.path.abspath(relative_path)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# Assuming the files are in the same directory as this script, load in the DS
script_dir = os.path.dirname(os.path.abspath(__file__))
input_ds_location = os.path.join(script_dir, "EFT_seed_data_input.jsonl")
input_dataset = Dataset.from_json(input_ds_location)

output_data = []

# Go thorugh dataset
def format_dataset(dataset, model, tokenizer):
    total_entries = len(dataset)
    for idx, row in enumerate(dataset):
        question = row["question"]
        answer = row["answer"]
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
User: {question}
<response>{answer}</response>
After examining the user’s instruction and the response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: “Score: <total points>”
Remember to assess from the AI Assistant perspective, utilizing web search knowledge as
necessary. To evaluate the response in alignment with this additive scoring model, we’ll
systematically attribute points based on the outlined criteria.
        '''
        model_inputs = tokenizer(prompt, return_tensors="pt")

        tokens = model_inputs["input_ids"].to(device)

        # Note the length of the input
        input_length = tokens.shape[1]

        generation_output = model.generate(
            tokens,
            # do_sample=True, # Picks tokens from the prob. distribution for more creative responses
            # temperature=0.7, # randomness in sampling (higher temp, more creative, but more random, lower, more predictable), effects logits
            # top_p=0.9, # Limits the set of posible next tokens. Does so by cumulatively selecting the most probable tokens from a prob. distribution until it reaches the limit
            # top_k=20,   # Limits the options to the {top_k} most likely options
            max_new_tokens=115, # Output max length
            pad_token_id=tokenizer.eos_token_id
        )
        # Decode only the newly generated tokens, ignoring the input part
        # Subtract input_length from the generated_ids' length to get only new tokens
        new_tokens = generation_output[0, input_length:].tolist()  # Get only the new token ids
        output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # TODO include rank
        output_data.append((prompt, output))
        print(f"Processing: {idx+1}/{total_entries}")
    return pd.DataFrame(output_data, columns=['prompt', 'response']) # Return as prompt, response

output_df = format_dataset(input_dataset, model, tokenizer)
output_file_path = os.path.join(script_dir, "EFT_seed_data.jsonl")
output_df.to_json(output_file_path, orient="records", lines=True)