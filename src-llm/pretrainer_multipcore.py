import logging
from components.gpt_model import GPTModel
from components.config import GPT_CONFIG_124M
from components.data_loader import create_dataloader
from accelerate import Accelerator
from torch.optim import AdamW
import torch

logger = logging.getLogger(__name__)
accelerator = Accelerator()

batch_size = GPT_CONFIG_124M["batch_size"]
block_size = GPT_CONFIG_124M["ctx_len"]
learning_rate = GPT_CONFIG_124M["learning_rate"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = GPT_CONFIG_124M["eval_iters"]
max_iters = GPT_CONFIG_124M["max_iters"]
eval_interval = GPT_CONFIG_124M["eval_interval"]

def main():
    def get_batch(dataloader):
        for batch in dataloader:
            X, Y = batch
            return X, Y
    #loss
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

    device = accelerator.device
    with open("src-llm/components/instructions.txt", "r", encoding="utf-8") as f:
            text = f.read()
    training_data = create_dataloader(text, batch_size=8, max_length=4, stride=5)
    val_data = create_dataloader(text, batch_size=8, max_length=4, stride=5)
    model = GPTModel(GPT_CONFIG_124M).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model, optimizer, training_data, val_data= accelerator.prepare(
        model, optimizer, training_data, val_data 
    )

    logger.info('traing begins')
    for iter in range(500):
        # print(f"training begins")
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model,training_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # # sample a batch of data
        xb, yb = get_batch(training_data)
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()

    logger.info('training finished')    
if __name__ == "__main__":
    main()