import tiktoken
import torch
from config import GPT_CONFIG_124M
from gpt_model import GPTModel
from torch.optim import AdamW
import logging
from data_loader import create_dataloader

batch_size = GPT_CONFIG_124M["batch_size"]
block_size = GPT_CONFIG_124M["ctx_len"]
learning_rate = GPT_CONFIG_124M["learning_rate"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
max_iters = GPT_CONFIG_124M["max_iters"]
eval_interval = GPT_CONFIG_124M["eval_interval"]



# data loading
def get_batch(split):
    for batch in dataloader:
        xb, yb = batch
        return xb.to(device), yb.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":

    with open("src-llm/components/instructions.txt", "r", encoding="utf-8") as f:
        text = f.read()

    dataloader = create_dataloader(text, batch_size=8, max_length=4, stride=5)

    print(dataloader)




    # intialize model 
    model = GPTModel(GPT_CONFIG_124M).to(device)

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # training loop 
    # optimizer = Adam(model.parameters())

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
        xb, yb = get_batch(dataloader)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # genarate a sequence of tokens
    # context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # print(tokenizer.decode(model.generate(context, max_new_tokens=1000)[0].tolist()))
    # save the model
    # torch.save(model.state_dict(), 'model.pt')\
    # model.predict("SELECT * FROM users WHERE name = 'John'")
    

