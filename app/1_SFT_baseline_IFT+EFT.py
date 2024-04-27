import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset, concatenate_datasets
import os

# TODO: figure out how to continue fine-tuning on top of the already fine-tuned IFT adapter

def get_model_and_tokenizer(model_name_or_path):
    """ Initialize and return the model and tokenizer with specific configuration. """
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",# for higher precision
        bnb_4bit_use_double_quant=True, # Helpes memory usage
        bnb_4bit_compute_dtype=torch.bfloat16, # Should provide faster learning
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token # Have to set this otherwise it crashes
    return model, tokenizer

# TODO: This function seems a bit too specific
def get_tokenized_ds(file_path, tokenizer, max_length=1024):
    dataset = Dataset.from_json(file_path)
    tokenized_dataset = dataset.map(
        lambda example: tokenizer(example["text"], max_length=max_length, truncation=True, padding="max_length"), 
        batched=True
    )
    return tokenized_dataset

# TODO: Temp function to combine 2 datasets and tokenize them
def get_tokenized_ds2(train_file_ds_1, train_file_ds_2, tokenizer):
    max_length = 1024

    # Load datasets from JSON files
    dataset_1 = Dataset.from_json(train_file_ds_1)
    dataset_2 = Dataset.from_json(train_file_ds_2)

    # Process dataset_2: merge 'prompt' and 'response' fields into a single 'text' field
    dataset_2 = dataset_2.map(lambda example: {'text': f"{example['prompt']} {example['response']}"})

    # Optionally, remove other fields if dataset_2 contains more fields than 'prompt' and 'response'
    # This is not strictly necessary if the map function only returns the new 'text' field
    dataset_2 = dataset_2.remove_columns([col for col in dataset_2.column_names if col not in ['text']])

    # Append dataset_1 to dataset_2
    combined_dataset = concatenate_datasets([dataset_1, dataset_2])
    
    # output the non tokenized ds
    output_jsonl_path = 'combined_dataset.jsonl'  # Define the path where the file will be saved
    combined_dataset.to_json(output_jsonl_path, orient='records', lines=True)
    print(f"Dataset saved to {output_jsonl_path}")

    # Tokenize the combined dataset
    tokenized_dataset = combined_dataset.map(
        lambda example: tokenizer(example['text'], max_length=max_length, truncation=True, padding="max_length"),
        batched=True
    )
    output_jsonl_path = 'combined_dataset2.jsonl'  # Define the path where the file will be saved
    tokenized_dataset.to_json(output_jsonl_path, orient='records', lines=True)
    return tokenized_dataset

def get_lora_configured_model(model):
    """ Defines and returns a peft model with the LoRA config """
    lora_config = LoraConfig(
        r=16, #As bigger the R bigger the parameters to train.
        lora_alpha=16, #default 16, should not be modified. a scaling factor that adjusts the magnitude of the weight matrix. It seems that as higher more weight have the new training.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"], # All layers for mistral "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"
        lora_dropout=0.05, #Helps to avoid Overfitting. #TODO how does it do that?
        bias="none",  # this specifies if the bias parameter should be trained.
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model

def get_trainer(model, tokenizer, train_dataset, output_dir):
    """ Gets the trainer class with all of the paramaters defined """
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        # per_device_eval_batch_size=1, # No eval
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
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
    model, tokenizer = get_model_and_tokenizer(model_name_or_path)

    # Preparing model for training 
    # TODO: Figure out what these paramaters do, without them the training runs out of memory
    model.gradient_checkpointing_enable() 
    model = prepare_model_for_kbit_training(model)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")

    # TODO: maybe this merge will not be necessary /Temp code
    train_file_ds = os.path.join(script_dir, "datasets/0_IFT_seed_data.jsonl")
    train_file_ds_2 = os.path.join(script_dir, "datasets/EFT/EFT_seed_data_final.jsonl")
    train_ds_tokenized = get_tokenized_ds2(train_file_ds, train_file_ds_2, tokenizer)
    # /Temp code

    # Previous version with just IFT
    # train_file_ds = os.path.join(script_dir, "../dataset/0_IFT_seed_data.jsonl")
    # train_ds_tokenized = get_tokenized_ds(train_file_ds, tokenizer)

    model = get_lora_configured_model(model)
    model.print_trainable_parameters()
    model.config.use_cache = False  # disable cache to prevent warning, re-enable for inference if needed

    trainer = get_trainer(model, tokenizer, train_ds_tokenized, output_dir)
    trainer.train()

    output_file_name = "/Mistral-7B-Instruct-v0.2-SFT_baseline_IFT+EFT"
    model.save_pretrained(output_dir + output_file_name)
    tokenizer.save_pretrained(output_dir + output_file_name)

if __name__ == "__main__":
    main()