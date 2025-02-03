# Import required libraries
import json
import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, Trainer
from data_preprocessing import preprocess_data

# Load configuration
with open("./config/config.json") as f:
    config = json.load(f)

# Load dataset
dataset, classes = preprocess_data()

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("./models/distilbert-finetuned-emotion").to(device)

# Evaluate model
trainer = Trainer(model = model)
pred_outputs = trainer.predict(dataset["test"])
y_preds = pred_outputs.predictions.argmax(-1)
y_true = dataset["test"][:]["label"]

print("Classification Report")
print(classification_report(y_true, y_preds, target_names = classes))
