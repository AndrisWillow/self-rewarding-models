import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
import os

# TODO: figure out how to continue fine-tuning on top of the already fine-tuned IFT adapter