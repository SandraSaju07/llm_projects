# Import required libraries
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load configuration
with open("./config/config.json") as f:
    config = json.load(f)

# Load tokenizer & model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
model = AutoModelForSequenceClassification.from_pretrained("./models/distilbert-finetuned-emotion").to(device)

# Prediction function
def predict_text(text):
    input_encoded = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**input_encoded)
    pred = torch.argmax(outputs.logits, dim = 1).item()
    return pred

# Main Function to execute an Example
if __name__ == "__main__":
    text = "I love the world"
    prediction = predict_text(text)
    print(f"Predicted Emotion is: {prediction}")
