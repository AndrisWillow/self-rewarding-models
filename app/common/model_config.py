
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
def get_model_and_tokenizer(model_name_or_path):
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=config)
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    return model, tokenizer