import tiktoken
import torch
from config import GPT_CONFIG_124M
from gpt_model import GPTModel
from torch.optim import AdamW
import logging

batch_size = 16
block_size = GPT_CONFIG_124M["ctx_len"]
# batch_size = GPT_CONFIG_124M["batch_size"]
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
max_iters = 1000
eval_interval = 100



# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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


    with open("/Users/jagpreetsingh/ML_Projects/text-sql/src-llm/components/instructions.txt", "r", encoding="utf-8") as f:
        text = f.read()



    # Tokenization
    tokenizer = tiktoken.get_encoding("gpt2")


    # Train and test splits
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:] 




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
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # save the model
    torch.save(model.state_dict(), 'model.pt')
    

