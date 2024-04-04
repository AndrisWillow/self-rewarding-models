from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer 
from datasets import load_dataset
import torch

device = "cuda"

# TODO add in context learning?

# Load dataset
dataset = load_dataset('ai2_arc', 'ARC-Challenge', split="test")
# Load model
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ" #TheBloke/Mistral-7B-Instruct-v0.2-AWQ #TheBloke/Mistral-7B-v0.1-AWQ
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                          trust_remote_code=False, safetensors=True)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

Score = 0

# Go thorugh dataset
for row in dataset:
    question = row['question']
    answerKey = row['answerKey']
    choices_formatted = " ".join([f"{label}: {text}" for label, text in zip(row['choices']['label'], row['choices']['text'])])

    prompt=f'''Answer this multiple choice question. 
    Question: {question} Posible answers: {choices_formatted} Output only the corresponding letter to the correct answer. Answer:
    '''

    model_inputs = tokenizer(prompt, return_tensors="pt")

    tokens = model_inputs["input_ids"].to(device)

    # Note the length of the input
    input_length = tokens.shape[1]

    # Move the model to the specified device
    model.to(device)

    generation_output = model.generate(
        tokens,
        do_sample=True, # Picks tokens from the prob. distribution for more creative responses
        temperature=1, # randomness in sampling (higher temp, more creative, but more random, lower, more predictable), effects logits
        top_p=1, # Limits the set of posible next tokens. Does so by cumulatively selecting the most probable tokens from a prob. distribution until it reaches the limit
        top_k=1,   # Limits the options to the {top_k} most likely options
        max_new_tokens=1 # Output max length
    )
    # Decode only the newly generated tokens, ignoring the input part
    # Subtract input_length from the generated_ids' length to get only new tokens
    # print(input_length)
    new_tokens = generation_output[0, input_length:].tolist()  # Get only the new token ids
    # output = tokenizer.decode(generation_output[0])
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Compare output
    print(f'\n {prompt} Expected: {answerKey} Got: {output} Score: {Score}')
    if output == answerKey:
        Score += 1

Result = Score / dataset.num_rows
print(f'\n Result: {Result}')