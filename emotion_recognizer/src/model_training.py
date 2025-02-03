# Import required libraries
import json
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import preprocess_data

# Load configuration
with open("./config/config.json") as f:
    config = json.load(f)

# Load dataset
dataset, classes = preprocess_data()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(config["model_checkpoint"], num_labels = len(classes)).to(device)

# Training Arguments
training_args = TrainingArguments(
    output_dir = "./models/distilbert-finetuned-emotion",
    num_train_epochs = config["num_train_epochs"],
    learning_rate = config["learning_rate"],
    per_device_train_batch_size = config["batch_size"],
    per_device_eval_batch_size = config["batch_size"],
    weight_decay = config["weight_decay"],
    eval_strategy = "epoch",
    logging_dir = "./logs"
)

# Metrics calculation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy_score": accuracy_score(labels, preds),
        "f1_score": f1_score(labels, preds, average="weighted")
    }

# Trainer
trainer = Trainer(
    model = model,
    args = training_args,
    compute_metrics = compute_metrics,
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"]
)

# Train the model
if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./models/distilbert-finetuned-emotion")
    print("Training completed. Model got saved...")