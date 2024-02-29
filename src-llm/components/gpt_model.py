import sys
sys.path.append('components/')
import torch
import torch.nn as nn
from layer_norm import LayerNorm
from transfomerblock import TransformerBlock
from config import GPT_CONFIG_124M
import tiktoken
from torch.nn import functional as F

class GPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["ctx_len"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.block_size = cfg["ctx_len"]

        self.trf_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
            )
        
        self.final_norm = LayerNorm(cfg["emb_dim"],)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx, targets=None):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds # [batch_size, num_tokens, emb_size]
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        # return logits
        if targets is None:
            loss = None
        else:
            batch_size, seq_len, emb_size= logits.shape
            logits = logits.view(batch_size*seq_len, emb_size)
            targets = targets.view(batch_size*seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# if __name__ == "__main__":
#     batch = []
#     tokenizer = tiktoken.get_encoding("gpt2")


#     # batch.append(torch.tensor(tokenizer.encode(txt1)))
#     # batch.append(torch.tensor(tokenizer.encode(txt2)))
#     # batch = torch.stack(batch, dim=0)
#     # # print(batch)
    
#     # torch.manual_seed(123)

#     # model = GPTModel(GPT_CONFIG_124M)
#     # out = model(batch)
#     # print(f"Input batch:\n {batch}")
#     # print(f"Output shape:\n {out.shape}")
#     # print(out)