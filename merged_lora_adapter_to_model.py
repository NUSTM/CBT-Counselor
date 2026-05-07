import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_path = "/path/to/base_model"
lora_path = "/path/to/checkpoints/lora_adapter"
merged_path = "/path/to/checkpoints/merged_model"

if not os.path.exists(merged_path):
    os.mkdir(merged_path)

base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
model = PeftModel.from_pretrained(base_model, lora_path)
merged = model.merge_and_unload()

merged.save_pretrained(merged_path, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
tokenizer.save_pretrained(merged_path)