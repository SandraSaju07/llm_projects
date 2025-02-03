# Import required libraries
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
import numpy as np

# Load the tokenizer and trained model
MODEL_CKPT = "distilbert-base-uncased"
MODEL_PATH = "./models/distilbert-finetuned-emotion"

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

# Define class labels
classes = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Streamlit App UI
st.title("Emotion Recognizer")
st.write("Enter a text to analyze its emotion:")

# User Input
text_input = st.text_area("Enter your text here", "")

if st.button("Analyze Emotion"):
    
    # Tokenize the text input
    if text_input:
        input_encoded = tokenizer(text_input, return_tensors = "pt").to(device)

        # Predict emotion of the text input
        with torch.no_grad():
            outputs = model(**input_encoded)

        logits = outputs.logits
        pred = torch.argmax(logits, dim = 1).item()

        # Display results
        st.success(f"Predicted Emotion: **{classes[pred]}**")

    else:
        st.warning("Please enter some text")


