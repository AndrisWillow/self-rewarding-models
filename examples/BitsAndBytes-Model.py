from datasets import load_dataset
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=config, device_map="auto")

device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

# With adapter
# relative_path = 'outputs/Mistral-7B-Instruct-v0.2-SFT_baseline_IFT+EFT'
# adapter_path = os.path.abspath(relative_path)
# print("Absolute path:", adapter_path)
# model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

system_prompt =  """You are a helpful AI assitant that gives helpful, detailed, and polite answers to the user's questions. Answer the following user question only: """

prompts = (
    f"""{system_prompt}
    Human: I am using docker compose and i need to mount the docker socket - how would i do that?"
    Assistant: ,
    """,
    f"""{system_prompt}
    Human: How to make a simple game in Python?
    Assistant: 
    """,
    f"""{system_prompt}
    Human: Can you help me write a CV?
    Assistant: 
    """,
    f"""{system_prompt}
    Human: What is the boiling point of water in Celsius?
    Assistant: 
    """,
    f"""{system_prompt}
    Human: Play me a song
    Assistant: 
    """,
)
for prompt in prompts:
    model_inputs = tokenizer(prompt, return_tensors="pt")
    tokens = model_inputs["input_ids"].to(device)
        # Note the length of the input
    input_length = tokens.shape[1]
    generation_output = model.generate(
        tokens,
        # do_sample=True, # Picks tokens from the prob. distribution for more creative responses
        # temperature=1, # randomness in sampling (higher temp, more creative, but more random, lower, more predictable), effects logits
        # top_p=1, # Limits the set of posible next tokens. Does so by cumulatively selecting the most probable tokens from a prob. distribution until it reaches the limit
        # top_k=1,   # Limits the options to the {top_k} most likely options
        max_new_tokens = 250,
        # top_p = 0.9,
        # temperature=0.7,
        # do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    new_tokens = generation_output[0, input_length:].tolist()  # Get only the new token ids
    # output = tokenizer.decode(generation_output[0])
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f'\n {output}')