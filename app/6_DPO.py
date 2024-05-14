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

def load_model_and_tokenizer(model_name_or_path):
    """ Initialize the model and tokenizer with specific configurations. """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def preprocess_dataset(dataset, tokenizer):
    """ Preprocess dataset entries by applying tokenizer and modifications. """
    def get_prompt(example):
        prompt_sample = [{"role": "user", "content": example['prompt']}]
        prompt_for_model = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
        example['prompt'] = prompt_for_model
        example['chosen'] += tokenizer.eos_token
        example['rejected'] += tokenizer.eos_token
        return example

    return dataset.map(get_prompt)

# from https://github.com/mlabonne/llm-course/blob/main/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb
# Also from https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2
def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16, #As bigger the R bigger the parameters to train.
        lora_alpha=16, #default 16, should not be modified. a scaling factor that adjusts the magnitude of the weight matrix.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"], # All layers for mistral "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"
        lora_dropout=0.05, #Helps to avoid Overfitting. #TODO how does it do that?
        bias="none",  # this specifies if the bias parameter should be trained.
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    return model, peft_config

def get_trainer(base_model, tokenizer, lora_config, dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        learning_rate=5e-5,
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
    return trainer

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '../outputs/Mistral-7B-v0.1-SFT_baseline_IFT+EFT')
    dataset_file = "app/datasets/preference_pairs/preference_pairs.jsonl"
    output_dir = "outputs/Mistral-7B-v0.1-SFT_baseline-DPO-M1"

    base_model, tokenizer = load_model_and_tokenizer(model_path)
    base_model.config.use_cache = False  # To prevent caching during training

    dataset = Dataset.from_json(dataset_file)
    dataset = preprocess_dataset(dataset, tokenizer)

    model, lora_config = create_peft_config(base_model)

    DPO_trainer = get_trainer(model, tokenizer, lora_config, dataset, output_dir)
    DPO_trainer.train()
    DPO_trainer.model.save_pretrained(output_dir)

if __name__ == "__main__":
    main()