import chainlit as cl
import torch
from components.gpt_model import GPTModel
from components.config import GPT_CONFIG_124M
import tiktoken

# # intialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPTModel(GPT_CONFIG_124M).to(device)
model.load_state_dict(torch.load("src-llm/artifacts/model_2024-03-23_01-39-17.pt"))
tokenizer = tiktoken.get_encoding("gpt2")


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...

    prompt = message.content
    context = torch.tensor(
        tokenizer.encode(prompt, allowed_special={"<|startoftext|>", "<|endoftext|>"})
    ).view(1, -1)
    pred = tokenizer.decode(model.generate(context, max_new_tokens=100)[0].tolist())

    # Send a response back to the user
    await cl.Message(
        content=pred,
    ).send()
