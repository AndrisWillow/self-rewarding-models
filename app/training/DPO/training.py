# Expected DPO dataset output:

# Prompt, chosen, rejected

import torch
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1, # Note that the beta is the temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

