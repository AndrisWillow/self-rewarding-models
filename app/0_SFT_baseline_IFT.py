import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
import os

def get_model_and_tokenizer(model_name_or_path):
    """ Initialize and return the model and tokenizer with specific configuration. """
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",# for higher precision
        bnb_4bit_use_double_quant=True, # Helps memory usage
        bnb_4bit_compute_dtype=torch.bfloat16, # Should provide faster learning
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token # Have to set this otherwise it crashes
    return model, tokenizer

def get_tokenized_ds(file_path, tokenizer, max_length=1024):
    dataset = Dataset.from_json(file_path)
    tokenized_dataset = dataset.map(
        lambda example: tokenizer(example["text"], max_length=max_length, truncation=True, padding="max_length"), 
        batched=True
    )
    return tokenized_dataset

def get_lora_configured_model(model):
    """ Defines and returns a peft model with the LoRA config """
    lora_config = LoraConfig(
        r=16, #As bigger the R bigger the parameters to train.
        lora_alpha=16, #default 16, should not be modified. a scaling factor that adjusts the magnitude of the weight matrix.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"], # All layers for mistral "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"
        lora_dropout=0.05, #Helps to avoid Overfitting. #TODO how does it do that?
        bias="none",  # this specifies if the bias parameter should be trained.
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    return model

def get_trainer(model, tokenizer, train_dataset, output_dir):
    """ Gets the trainer class with all of the paramaters defined """
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1, # More than 1 epoch seems to overfit on the training data already
        weight_decay=0.01,
        learning_rate=1e-4, # Base learning rate, can posibly be unstable
        save_strategy="epoch",
        fp16=True,
        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_8bit"
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    return trainer

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = "outputs"
    model_name_or_path = "mistralai/Mistral-7B-v0.1"
    model, tokenizer = get_model_and_tokenizer(model_name_or_path)
    # Preparing model for training 
    # TODO: Figure out what these paramaters do, without them the training runs out of memory
    model.gradient_checkpointing_enable() 
    model = prepare_model_for_kbit_training(model)

    train_file_ds = os.path.join(script_dir, "datasets/IFT_seed_data/IFT_seed_data.jsonl")
    train_ds_tokenized = get_tokenized_ds(train_file_ds, tokenizer)

    model = get_lora_configured_model(model)
    model.print_trainable_parameters()
    model.config.use_cache = False  # disable cache to prevent warning, re-enable for inference if needed

    trainer = get_trainer(model, tokenizer, train_ds_tokenized, output_dir)
    trainer.train()

    output_file_name = "/Mistral-7B-v0.1-SFT_baseline_IFT"
    model.save_pretrained(output_dir + output_file_name)
    tokenizer.save_pretrained(output_dir + output_file_name)

if __name__ == "__main__":
    main()