# Import required libraries
import pandas as pd
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Load configuration
with open("./config/config.json") as f:
    config = json.load(f)

# Load dataset
def load_emotion_dataset():
    dataset = load_dataset("dair-ai/emotion")
    return dataset

# Tokenization function
def tokenize_function(batch):
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    return tokenizer(batch["text"], padding=True, truncation=True)

# Preprocess data
def preprocess_data():
    dataset = load_emotion_dataset()
    classes = dataset["train"].features["label"].names
    dataset = dataset.map(tokenize_function, batched=True, batch_size=None)
    return dataset, classes

# Main Function
if __name__ == "__main__":
    dataset, classes = preprocess_data()
    print("Data preprocessed successfully...")

