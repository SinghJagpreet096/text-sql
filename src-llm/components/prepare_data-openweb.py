from datasets import load_dataset


dataset = load_dataset("Bingsu/openwebtext_20p", )


# print(dataset['train']['text'][0:10])
sample = dataset['train']['text']
# print(sample)
# print(f"no of tokens: {len(dataset['train']['text'])}")


with open ("src-llm/data/train/webtext-20p.txt", "w") as f:
    for s in sample:
        f.write(s + "\n")