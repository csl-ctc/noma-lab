
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "FacebookAI/roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("Model loaded:", model_name)
