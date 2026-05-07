import os

root_dir = "xxxxx"


DIALOGPT_DIR = os.path.join(root_dir, "DialoGPT-medium")

# calculate BERTScore
BERT_DIR = os.path.join(root_dir, "bert-base-uncased")

# calculate toxic
TOXIC_ROBERTA = os.path.join(root_dir, "roberta_toxicity_classifier")


# calculate EPITOME score
EPITOME_SAVE_DIR = os.path.join(root_dir, "epitome_save_dir")

BERT_CONFIG_PATH = f"{root_dir}/bert-base-uncased/config.json"
ROBERTA_MODEL_DIR = f"{root_dir}/roberta-base/pytorch_model.bin"
ROBERTA_DIR = f"{root_dir}/roberta-base/"
ROBERTA_CONFIG_PATH = f"{root_dir}/roberta-base/config.json"
BERT_MODEL_DIR = f"{root_dir}/bert-base-uncased/pytorch_model.bin"