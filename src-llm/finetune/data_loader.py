import sys
sys.path.append('src-llm/components/')
import tiktoken
from gpt_model import GPTModel
from pretrainer import estimate_loss
from config import GPT_CONFIG_124M
from torch.optim import AdamW
import torch
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np

tokenizer = tiktoken.get_encoding("gpt2")
with open ("spider_train.txt", "r") as file:
    train = file.read()
with open ("spider_test.txt", "r") as file:
    test = file.read()



tokenised_train = tokenizer.encode(train,allowed_special={"<|endoftext|>", "<|startoftext|>"})
tokenised_test = tokenizer.encode(test,allowed_special={"<|endoftext|>", "<|startoftext|>"})


# print(f"train : {tokenised_train[0:10]}")
# print(f"test : {tokenised_test[:10]}")

# instantiate the model
model = GPTModel(GPT_CONFIG_124M)

# create optimizer
optimizer = AdamW(model.parameters(), lr=0.001)

model.load_state_dict(torch.load("model.pt"))

# model state dict
# print(model.state_dict().keys())

model.train()

training_args = TrainingArguments(output_dir="tunner", num_train_epochs=1, per_device_train_batch_size=4, per_device_eval_batch_size=4, warmup_steps=500, weight_decay=0.01, logging_dir="logs", logging_steps=10)
    
# print("training_args : ", training_args)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=GPT_CONFIG_124M,
    train_dataset=tokenised_train,
    eval_dataset=tokenised_test,
    compute_metrics=compute_metrics,
)

trainer.train()

