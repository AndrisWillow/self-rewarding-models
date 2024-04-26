import os
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model from hub
model_name_or_path = "TheBloke/Mistral-7B-v0.1-GPTQ" #TheBloke/Mistral-7B-Instruct-v0.2-AWQ #TheBloke/Mistral-7B-v0.1-AWQ TheBloke/Mistral-7B-v0.1-GPTQ

# What are fuse_layers, safetensors?
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto", #TODO Test with Cuda
                                            #  device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                             trust_remote_code=False, # prevents running custom model files on your machine
                                             revision="main") # which version of model to use in repo
model.eval()
device = "cuda"

# With adapter # Training seems to have done very little or notihng at all
# relative_path = "./trained-models/TheBloke-Mistral-7B-v0.1-GPTQ-OpenAssistantGuanaco2"
# adapter_path = os.path.abspath(relative_path)
# print("Absolute path:", adapter_path)
# model = PeftModel.from_pretrained(model, adapter_path)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

user_question = "I am using docker compose and i need to mount the docker socket - how would i do that?"

prompt = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "Human: i'm writing a dungeons and dragons campaign for a couple of friends, given the description of the hellbound city of Dis, could you come up with a couple of cool activities my players could do once they get there? remember that having fun and seeing cool things is the main focus, no matter how serious or absurd a situation can get. The City of Dis is incredibly massive. The walls, buildings, and other structures clutter the immense black metropolis and glow at the edges like heated glass. The city is a broken maze of metal ramparts built into miles of steep hillsides and shattered towers framing an endless labyrinth of twisting alleyways. The air is extremely hot and distorts with the heat as smoke rises from the streets themselves, with the occasional ash or ember drifting through the air. Most of the buildings are built out of dark stone, climbing for many stories. The architecture is hooked to the corner as the monolithic deep reds, blacks, and grays, nothing belies an exterior that would seem to be welcoming to a traveling pack of people. At the center, visible everywhere in the city, is the Iron Tower, an impossibly colossal and unreachable tower of black iron and lead that stretches for hundreds of miles into the sky before vanishing into the dark, clouded expanse. The Iron Tower is the throne of the archdevil Dispater and stands ever-vigilant over his Iron City of Dis. The city of Dis is always changing and being rebuilt in accordance with the decree of paranoid Dispater. ?"
    "Assistant: "
)

model_inputs = tokenizer(prompt, return_tensors="pt")

tokens = model_inputs["input_ids"].to(device)

# Note the length of the input
input_length = tokens.shape[1]

# Move the model to the specified device
model.to(device)

generation_output = model.generate(
    tokens,
    # do_sample=True, # Picks tokens from the prob. distribution for more creative responses
    # temperature=0.8, # randomness in sampling (higher temp, more creative, but more random, lower, more predictable), effects logits
    # top_p=0.95, # Limits the set of posible next tokens. Does so by cumulatively selecting the most probable tokens from a prob. distribution until it reaches the limit
    # top_k=40,   # Limits the options to the {top_k} most likely options
    max_new_tokens=255, # Output max length
    pad_token_id=tokenizer.eos_token_id
)
# Decode only the newly generated tokens, ignoring the input part
# Subtract input_length from the generated_ids' length to get only new tokens
# print(input_length)
new_tokens = generation_output[0, input_length:].tolist()  # Get only the new token ids
# output = tokenizer.decode(generation_output[0])
output = tokenizer.decode(new_tokens, skip_special_tokens=True)

# print output
print(f'\n {output}')