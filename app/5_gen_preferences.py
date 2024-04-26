import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
import os

# TODO: Create pairs of responses, one with higher score one with less, delete duplicates. Output DS in format: prompt, prompt_chosen_completion, prompt_rejected_completion, score_rejected, score_accepted 