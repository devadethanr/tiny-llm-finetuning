import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset

#loading the distilbert tokenizer
model_name = 'distilbert-base-uncased'
tokeniser = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

#loading the dataset from huggingface
dataset = load_dataset('imdb')
small_train_dataset = dataset['train'].shuffle(seed=42).select(range(1000))
print(f"small_train_dataset {small_train_dataset}")
small_test_dataset = dataset['test'].shuffle(seed=42).select(range(500))
print(f"small_test_dataset {small_test_dataset}")

#tokenising the dataset
def tokenise_function(examples):
    return tokeniser(examples['text'], padding = 'max_length', trucation = True)

tokenised_train_dataset = small_train_dataset.map(tokenise_function, batched=True)
print(f"tokenised_train_dataset {tokenised_train_dataset}")
tokenised_test_dataset = small_test_dataset.map(tokenise_function, batched=True)
print(f"tokenised_test_dataset {tokenised_test_dataset}")

#sample texts for inital inference
sample_texts = [
    "This movie was absolutely amazing!",
    "I found this movie to be quite boring.",
    "The acting was okay, but the story was confusing",
    "I loved all the twists in this movie"
]
inputs = tokeniser(sample_texts, padding=True, truncation=True, return_tensors='pt')
print(f"inputs {inputs}")

#initial inference
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    print(f"predictions {predictions}")
print("Initial predictions:")
for text, pred in zip(sample_texts, predictions):
    print(f"{text} - {'positive' if pred == 1 else 'negative'}")