import os
import shutil
from datasets import load_dataset


dataset = load_dataset(
    "Bingsu/openwebtext_20p",
)

sample = dataset["train"]["text"][:2000]
print("sample: ", sample[0:10])

shutil.rmtree("src-llm/data/train")
os.makedirs("src-llm/data/train")


with open("src-llm/data/train/webtext-20p.txt", "w") as f:
    for s in sample:
        f.write(s + "\n")