import torch
from peft import PeftModel
import transformers
import os, time
import tempfile
from transformers import OPTForCausalLM, AutoTokenizer


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
relative_path = 'outputs/Mistral-7B-Instruct-v0.2-SFT_baseline'
peft_model = os.path.abspath(relative_path)
LORA_WEIGHTS = peft_model

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
model = OPTForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload", 
)
    
model = PeftModel.from_pretrained(
    model, 
    LORA_WEIGHTS, 
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload", 

)

model = model.merge_and_unload()
model.save_pretrained("./outputs/Mistral-7B-Instruct-v0.2-SFT_baseline_merged")