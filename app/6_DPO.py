# Taken from https://github.com/Oxen-AI/Self-Rewarding-Language-Models/blob/main/scripts/05_dpo.py

from datasets import load_dataset, Dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import DPOTrainer
import os

# TODO: Refactor this code into more readable functions
script_dir = os.path.dirname(os.path.abspath(__file__))
model_name = os.path.join(script_dir, '../outputs/Mistral-7B-Instruct-v0.2-SFT_baseline_IFT+EFT')
base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
dataset_file = "datasets/preference_pairs/preference_pairs.jsonl"
output_dir = "outputs"

# load the training dataset
dataset = Dataset.from_json(dataset_file)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

device_map = "auto"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=False,
)
base_model.config.use_cache = False # TODO: What does this do?

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=False)
tokenizer.pad_token = tokenizer.eos_token

# TODO: evaluate if it's worth adding the chat_template
def get_prompt(example):
    prompt_sample = [
        {"role": "user", "content": example['prompt']}
    ]

    prompt_for_model = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
    example['prompt'] = prompt_for_model

    example['chosen'] = example['chosen'] + tokenizer.eos_token
    example['rejected'] = example['rejected'] + tokenizer.eos_token

    return example

dataset = dataset.map(get_prompt)

# from https://github.com/mlabonne/llm-course/blob/main/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb
lora_dropout=0.05
lora_alpha=16
lora_r=16
learning_rate=5e-5

batch_size = 4

def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        lora_dropout=lora_dropout,
        lora_alpha=lora_alpha,
        r=lora_r,
        bias="none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    return model, peft_config

model, lora_config = create_peft_config(base_model)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,

    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    warmup_steps=50,
    logging_steps=1,
    num_train_epochs=1,
    save_steps=50,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
)

trainer = DPOTrainer(
    base_model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    beta=0.1,
    max_prompt_length=1024,
    max_length=1536,
)

trainer.train()

output_dir = os.path.join(output_dir, "Mistral-7B-Instruct-v0.2-SFT_baseline-DPO-M1")
trainer.model.save_pretrained(output_dir)