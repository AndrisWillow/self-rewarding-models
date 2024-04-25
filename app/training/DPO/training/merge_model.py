from datasets import load_dataset
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

# to merge a model you need to dequantize it, which requires too much v-ram

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

relative_path = 'outputs/Mistral-7B-Instruct-v0.2-SFT_baseline'
peft_model = os.path.abspath(relative_path)
model = PeftModel.from_pretrained(base_model, peft_model)

merged_model = model.merge_and_unload()

merged_model.save_pretrained("./outputs/Mistral-7B-Instruct-v0.2-SFT_baseline_merged")
tokenizer.save_pretrained("./outputs/Mistral-7B-Instruct-v0.2-SFT_baseline_merged")