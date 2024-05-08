import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
import os

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # for higher precision
    bnb_4bit_use_double_quant=True, # Helpes memory usage
    bnb_4bit_compute_dtype=torch.bfloat16, #Should provide faster learning
) 

model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=config, device_map="auto") 
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

# Have to do this otherwise it crashes
tokenizer.pad_token = tokenizer.eos_token


# Assuming the files are in the same directory as this script, load in the DS
script_dir = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(script_dir, "datasets/0_IFT_seed_data.jsonl")

train_ds = Dataset.from_json(train_file_path)

# Tokenize the DS, truncate it for max_size due to memory constraints
max_length = 1024 
tokenizer.pad_token = tokenizer.eos_token
train_ds_tokenized = train_ds.map(lambda example: tokenizer(example["text"], max_length=max_length, truncation=True, padding="max_length"), batched=True)

# Preparing model for training
model.gradient_checkpointing_enable() 
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16, #As bigger the R bigger the parameters to train.
    lora_alpha=16, #default 16, should not be modified. a scaling factor that adjusts the magnitude of the weight matrix. It seems that as higher more weight have the new training.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"], # All layers for mistral "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"
    lora_dropout=0.05, #Helps to avoid Overfitting. #TODO how does it do that?
    bias="none",  # this specifies if the bias parameter should be trained.
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
# LoRA adds new hidden layers to train, it doesn't need to train a massive amount of paramaters. This shows the trainable parameter count
model.print_trainable_parameters()
batch_size = 1
num_epochs= 1 # Over 1 epoch it seems that the model starts over-fitting already
trainer = Trainer(
    model=model,
    train_dataset=train_ds_tokenized,
    args=TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        weight_decay=0.01,      
        learning_rate=1e-4,
        save_strategy="epoch",
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. re-enable for inference!

trainer.train()

model.save_pretrained("./outputs/Mistral-7B-Instruct-v0.2-SFT_baseline")
tokenizer.save_pretrained("./outputs/Mistral-7B-Instruct-v0.2-SFT_baseline")