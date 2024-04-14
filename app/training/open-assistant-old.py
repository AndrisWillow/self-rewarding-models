from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import transformers

# Training seems to have failed

model_name_or_path = "TheBloke/Mistral-7B-v0.1-GPTQ" #TheBloke/Mistral-7B-Instruct-v0.2-AWQ #TheBloke/Mistral-7B-v0.1-AWQ TheBloke/Mistral-7B-v0.1-GPTQ

# What are fuse_layers, safetensors?
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto", #TODO Test with Cuda
                                            #  device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                             trust_remote_code=False, # prevents running custom model files on your machine
                                             revision="main") # which version of model to use in repo

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                          trust_remote_code=False,
                                          use_fast=True ) #If available will use a Rust based tokenizer instead of Python
# LoRA config #TODO explain all the pramaters
# config = LoraConfig(
#     r=8, # Lora attention dimension (the "rank").
#     lora_alpha=32, # The alpha parameter for Lora scaling.
#     target_modules=["q_proj"], # Specific to the model, in this case targeting what? layers
#     lora_dropout=0.05, #
#     bias="none", #
#     task_type="CAUSAL_LM", #
# )

print(model)

config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"], # It should be decided which layers to target, in the paper() they chose query and value
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA trainable version of model
model.enable_input_require_grads() #TODO test if this is still necessary
model = get_peft_model(model, config)

# LoRA adds new hidden layers to train, it doesn't need to train a massive amount of paramaters. This shows the trainable parameter count
model.print_trainable_parameters()

dataset_name = "timdettmers/openassistant-guanaco"
# train_ds = load_dataset(dataset_name, split="train").select(range(100)) # train the first 100 data, to save time
# validation_ds = load_dataset(dataset_name, split="test").select(range(30)) # just use the first 30 data for evaluation
train_ds = load_dataset(dataset_name, split="train")
validation_ds = load_dataset(dataset_name, split="test")

# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["text"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )
    return tokenized_inputs

# tokenize training and validation datasets
tokenized_data_train = train_ds.map(tokenize_function, batched=True)
tokenized_data_test = validation_ds.map(tokenize_function, batched=True)

# setting pad token
tokenizer.pad_token = tokenizer.eos_token

# data collator, for dynamic padding of inputs
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

# hyperparameters
lr = 2e-4
batch_size = 4
num_epochs = 10

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir= "./trained-models",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=False, # Re-enable for A-100
    optim="paged_adamw_8bit",
)     

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data_train,
    eval_dataset=tokenized_data_test,
    args=training_args,
    data_collator=data_collator
)

model.config.use_cache = False  # silence the warnings. Only for training
trainer.train()

model.save_pretrained("./TheBloke-Mistral-7B-v0.1-GPTQ-OpenAssistantGuanaco")
tokenizer.save_pretrained("./TheBloke-Mistral-7B-v0.1-GPTQ-OpenAssistantGuanaco")