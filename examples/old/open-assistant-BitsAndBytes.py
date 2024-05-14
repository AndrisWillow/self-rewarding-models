import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset

# TODO expalin config in detail what exactlly it does
# The compute dtype is used to change the dtype that will be used during computation. For example, hidden states could be in float32 but computation can be set to bf16 for speedups. By default, the compute dtype is set to float32

# The 4bit integration comes with 2 different quantization types: FP4 and NF4. The NF4 dtype stands for Normal Float 4 and is introduced in the QLoRA paper
# You can switch between these two dtype using bnb_4bit_quant_type from BitsAndBytesConfig. By default, the FP4 quantization is used.

# We also advise users to use the nested quantization technique. This saves more memory at no additional performance - from our empirical observations, 
# this enables fine-tuning llama-13b model on an NVIDIA-T4 16GB with a sequence length of 1024, batch size of 1 and gradient accumulation steps of 4.
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # for higher precision
    bnb_4bit_use_double_quant=True, # Helpes memory usage
    bnb_4bit_compute_dtype=torch.bfloat16, #Should provide faster learning
) 

model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=config, device_map="auto") # device_map="auto" decides how to automatically distribute the model between hardware
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

# Check where the model was loaded in
print(f'Model loaded in: {model.hf_device_map}')

# Preparing dataset
dataset_name = "timdettmers/openassistant-guanaco"
train_ds = load_dataset(dataset_name, split="train")
validation_ds = load_dataset(dataset_name, split="test")

# Tokenizing dataset and also truncating to a maximum size due to memory constraints
max_length = 256 #512 #1024
# setting pad token
tokenizer.pad_token = tokenizer.eos_token
train_ds_tokenized = train_ds.map(lambda example: tokenizer(example["text"], max_length=max_length, truncation=True, padding="max_length"), batched=True)
validation_ds_tokenized = validation_ds.map(lambda example: tokenizer(example["text"],  max_length=max_length, truncation=True, padding="max_length"), batched=True)

# Preparing model for training
# TODO what does this do?
model.gradient_checkpointing_enable() 
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16, #As bigger the R bigger the parameters to train.
    lora_alpha=16, #default 16, should not be modified. a scaling factor that adjusts the magnitude of the weight matrix.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"], # All layers for mistral "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"
    lora_dropout=0.05, #Helps to avoid Overfitting. #TODO how does it do that?
    bias="none",  # this specifies if the bias parameter should be trained.
    task_type="CAUSAL_LM"
)

# Getting model ready for training
model = get_peft_model(model, config)
# LoRA adds new hidden layers to train, it doesn't need to train a massive amount of paramaters. This shows the trainable parameter count
model.print_trainable_parameters()

# Had to reduce batch-size because of memory constraints due to the dataset size. Batching seems to drastically increese the training time
batch_size = 4 # for 256 # 2 batches for 512, 1 batch 1024
num_epochs= 1 # Over 1 epoch it seems that the model starts over-fitting already

# TODO what do these paramaters effect?
trainer = Trainer(
    model=model,
    train_dataset=train_ds_tokenized,
    eval_dataset=validation_ds_tokenized,
    args=TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        # warmup_steps=2, 
        # max_steps=10, # Can be used instead of epochs, # TODO what is optimal amount of steps?        
        learning_rate=1e-4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        load_best_model_at_end=True,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Re-enable for inference!

trainer.train()

model.save_pretrained("./outputs/Mistral-7B-Instruct-v0.2-openassistant-guanaco")
tokenizer.save_pretrained("./outputs/Mistral-7B-Instruct-v0.2-openassistant-guanaco")