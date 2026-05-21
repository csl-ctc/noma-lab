import os
from pathlib import Path

print("HF_ENDPOINT is set:", bool(os.environ.get("HF_ENDPOINT")))
print("HF_HUB_ETAG_TIMEOUT:", os.environ.get("HF_HUB_ETAG_TIMEOUT"))

os.environ["HF_HUB_DISABLE_XET"] = "1"

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

MODEL_ID = "distilbert-base-uncased"
DATASET_ID = "imdb"

OUTPUT_DIR = Path("trained_model")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading dataset from Hugging Face...")
dataset = load_dataset(DATASET_ID)

# GitHub Actions上では重くなりすぎないようにPoC用に件数を絞る
train_dataset = dataset["train"].shuffle(seed=42).select(range(100))
eval_dataset = dataset["test"].shuffle(seed=42).select(range(50))

print("Loading tokenizer from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    force_download=True,
)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )

print("Tokenizing dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("Loading model from Hugging Face...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    force_download=True,
)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)


print("Start training...")
trainer.train()

print("Saving trained model...")
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("Training completed.")
print("Saved to:", OUTPUT_DIR.resolve())
