from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

model_name = "bert-base-uncased"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model and tokenizer loaded successfully!")