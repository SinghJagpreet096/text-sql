import tiktoken
import torch
from config import GPT_CONFIG_124M
from gpt_model import GPTModel
from torch.optim import AdamW
import logging
from data_loader import create_dataloader
from datetime import datetime
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

batch_size = GPT_CONFIG_124M["batch_size"]
block_size = GPT_CONFIG_124M["ctx_len"]
learning_rate = GPT_CONFIG_124M["learning_rate"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = GPT_CONFIG_124M["eval_iters"]
max_iters = GPT_CONFIG_124M["max_iters"]
eval_interval = GPT_CONFIG_124M["eval_interval"]



# data loading
def get_batch(dataloader):
    for batch in dataloader:
        X, Y = batch
        return X.to(device), Y.to(device)
    

@torch.no_grad()
def estimate_loss(model,train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        if split == 'train':
            data = train_data
        else:
            data = val_data
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":

    with open("src-llm/data/train/raw_data.txt", "r", encoding="utf-8") as f:
        text = f.read()
    logging.info(f"no. of tokens: {len(text)}")
    n = int(0.9*len(text))

    train_data = create_dataloader(text[:n], batch_size=8, max_length=4, stride=5)
    val_data = create_dataloader(text[n:], batch_size=8, max_length=4, stride=5)

    # for batch in train_data:
    #     print(batch)
    #     break
    
    # # intialize model 
    model = GPTModel(GPT_CONFIG_124M).to(device)

    # # initialize optimizer    
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # training loop

    for iter in range(1000):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model,train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # # sample a batch of data
        xb, yb = get_batch(train_data)
    #     # print(type(xb), type(yb))

    #     # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # genarate a sequence of tokens
    tokenizer = tiktoken.get_encoding('gpt2')
    prompt = "s"
    context = torch.tensor(tokenizer.encode(prompt, allowed_special={"<|startoftext|>","<|endoftext|>"})).view(1, -1)
    print(tokenizer.decode(model.generate(context, max_new_tokens=100)[0].tolist()))
    
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
    

    # save the model
    torch.save(model.state_dict(), f"src-llm/artifacts/model_{current_datetime}.pt")

