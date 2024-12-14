from transformers import Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from fine_tune import model, tokenised_train_dataset, tokenised_test_dataset

# 1. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",  # change this from "tensorboard" to "none" or any other tracking tool if you prefer
    disable_tqdm=False,
)


# 2. Define the metric
def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  return {"accuracy": accuracy_score(labels, predictions)}


# 3. Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenised_train_dataset,
    eval_dataset=tokenised_test_dataset,
    compute_metrics=compute_metrics,
)

# 4. Train the model
trainer.train()

# 5. Save the fine-tuned model
trainer.save_model("./fine_tuned_model")