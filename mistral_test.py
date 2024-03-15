from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

model_name_or_path = "TheBloke/Mistral-7B-v0.1-AWQ"

# Load model
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                          trust_remote_code=False, safetensors=True)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

prompt = "Talk about science in a funny way"
prompt_template=f'''{prompt}

'''

print("\n\n*** Generate:")

tokens = tokenizer( 
    prompt_template,
    return_tensors='pt' # return as PyTorch tensor
).input_ids.cuda()


# Generate output/ What is tempeture?
generation_output = model.generate(
    tokens,
    do_sample=True, # Picks tokens from the prob. distribution for more creative responses
    temperature=0.7, # randomness in sampling (higher temp, more creative, but more random, lower, more predictable), effects logits
    top_p=0.95, # Limits the set of posible next tokens. Does so by cumulatively selecting the most probable tokens from a prob. distribution until it reaches the limit
    top_k=40,   # Limits the options to the {top_k} most likely options
    max_new_tokens=512 # Output max length
)

print("Output: ", tokenizer.decode(generation_output[0]))