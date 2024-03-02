import torch
import pandas as pd
import sys
sys.path.append('src-llm/')
from components.data_loader import create_dataloader
from components.gpt_model import GPTModel
from components.config import GPT_CONFIG_124M
import logging
import tiktoken
from components.pretrainer import get_batch, estimate_loss

import random
random.seed(0)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

allowed_special = {"<|startoftext|>","<|endoftext|>"}
n_freeze = 1
batch_size = GPT_CONFIG_124M["batch_size"]
block_size = GPT_CONFIG_124M["ctx_len"]
learning_rate = GPT_CONFIG_124M["learning_rate"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
max_iters = GPT_CONFIG_124M["max_iters"]
eval_interval = GPT_CONFIG_124M["eval_interval"]


def param_count(m):
    params = sum([p.numel() for p in m.parameters()])/1_000_000
    trainable_params = sum([p.numel() for p in m.parameters() if p.requires_grad])/1_000_000
    print(f"Total params: {params:.2f}M, Trainable: {trainable_params:.2f}M")
    return params, trainable_params


with open ("spider_train.txt", "r") as f:
    data = f.read()

dataloader = create_dataloader(data, batch_size=8, max_length=4, stride=5)

for batch in dataloader:
    print(batch)
    break
model = GPTModel(GPT_CONFIG_124M).to(device)
model.load_state_dict(torch.load('model.pt'))
logging.info("Model loaded")
print("Model loaded")


print("Model parameters:")
params, trainable_params = param_count(model)

# freeze layers (disable gradients)
for param in model.parameters(): param.requires_grad = False
for param in model.out_head.parameters(): param.requires_grad = True
for param in model.trf_block[n_freeze].parameters(): param.requires_grad = True

print("Model parameters after freeze weights:")
params, trainable_params = param_count(model)


# initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=GPT_CONFIG_124M["learning_rate"])

# training loop
# model.train()
for iter in range(100):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model,dataloader)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# # sample a batch of data
    xb, yb = get_batch(dataloader)
#     # print(type(xb), type(yb))

#     # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # genarate a sequence of tokens
tokenizer = tiktoken.get_encoding('gpt2')
prompt = "Question: How many heads of the departments are older than 56 ?\n'<|startoftext|>"
context = torch.tensor(tokenizer.encode(prompt,allowed_special=allowed_special)).view(1, -1)
simple_generate = (tokenizer.decode(model.generate(context, max_new_tokens=100)[0].tolist()))
print("\n\n")
query_generator = (tokenizer.decode(model.generate_query(context, max_new_tokens=100,end_token = tokenizer.encode("<|endoftext|>",allowed_special=allowed_special))[0].tolist()))

print("Simple generate: \n", simple_generate)
print("Simple generate length: ", len(simple_generate)," \n\n")
print("Query generate: \n", query_generator)
print("Query generate length:", len(query_generator)," \n\n")
