import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatset(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        ## tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        ## use a sliding window to chunk the book into overlapping sequence of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk =token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader(txt, batch_size=4, max_length=256, stride=128,shuffle=True):
    # intialize tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')

    # create dataset
    dataset = GPTDatset(txt, tokenizer, max_length, stride)

    # create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
