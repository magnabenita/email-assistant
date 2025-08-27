from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("csv", data_files={"train": "enron_dataset/politeness.csv"})

# Load T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Preprocessing function
def preprocess(batch):
    # Prefix "paraphrase: " for T5 task
    inputs = ["paraphrase: " + text for text in batch['original']]
    targets = batch['polite']  # target paraphrase

    # Tokenize inputs & outputs
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Apply preprocessing
dataset = dataset.map(preprocess, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

training_args = TrainingArguments(
    output_dir="./models/paraphraser",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch",       # keep if supported
    logging_steps=10,
    save_total_limit=2,
    fp16=False,
    learning_rate=5e-5
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train']
)

# Train
trainer.train()

# Save model & tokenizer
model.save_pretrained("./models/paraphraser")
tokenizer.save_pretrained("./models/paraphraser")

print("Paraphraser training completed!")
