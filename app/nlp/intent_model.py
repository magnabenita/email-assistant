# app/nlp/intent_model.py

import os
from typing import List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from datasets import load_dataset, Dataset
import pandas as pd

# -------------------------------
# Config
# -------------------------------
TRAIN_CSV = "enron_dataset/intent_labeled.csv"
MODEL_DIR = "models/intent_model"
NUM_EPOCHS = 3
BATCH_SIZE = 16

# Default intents (for zero-shot fallback)
INTENT_LABELS = [
    "request",
    "information",
    "greeting",
    "complaint",
    "other"
]

# -------------------------------
# Helper: Train BERT intent model
# -------------------------------
def train_intent_model(csv_file: str, model_dir: str):
    print("📥 Loading labeled dataset...")
    df = pd.read_csv(csv_file)
    
    # Ensure 'body' and 'label' columns exist
    if "body" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'body' and 'label' columns")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    
    num_labels = df['label'].nunique()
    
    print(f"🧠 Training BERT for {num_labels} classes...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    )
    
    # Tokenization function
    def tokenize(batch):
        return tokenizer(batch['body'], padding=True, truncation=True)
    
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Split train/test
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))
    
    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"✅ Model trained and saved to {model_dir}")
    
    return model, tokenizer

# -------------------------------
# Load or fallback
# -------------------------------
if os.path.exists(TRAIN_CSV):
    try:
        model, tokenizer = train_intent_model(TRAIN_CSV, MODEL_DIR)
        print("✅ Using fine-tuned BERT model")
        
        def predict_intent(email_text: str) -> dict:
            inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = int(torch.argmax(logits, dim=1))
            return {
                "email_text": email_text,
                "predicted_intent": predicted_class
            }
        
    except Exception as e:
        print("⚠️ Training failed, fallback to zero-shot:", e)
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        def predict_intent(email_text: str, candidate_labels: List[str] = INTENT_LABELS) -> dict:
            result = classifier(email_text, candidate_labels)
            return {
                "email_text": email_text,
                "predicted_intent": result['labels'][0],
                "scores": dict(zip(result['labels'], result['scores']))
            }
else:
    print("⚠️ Labeled CSV not found, using zero-shot model")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    def predict_intent(email_text: str, candidate_labels: List[str] = INTENT_LABELS) -> dict:
        result = classifier(email_text, candidate_labels)
        return {
            "email_text": email_text,
            "predicted_intent": result['labels'][0],
            "scores": dict(zip(result['labels'], result['scores']))
        }

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    sample_email = "Hi team, can you send me the report by tomorrow?"
    prediction = predict_intent(sample_email)
    print(prediction)
