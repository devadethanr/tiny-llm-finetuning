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