from datasets import load_dataset


dataset = load_dataset("Bingsu/openwebtext_20p", )


# print(dataset['train']['text'][0:10])
text = dataset['train']['text']
# print(sample)
# print(f"no of tokens: {len(dataset['train']['text'])}")
no_of_tokens = len(text)


with open ("src-llm/data/train/webtext-20p.txt", "w") as f:
    for s in text[:int(no_of_tokens/5)]:
        f.write(s + "\n")
print("done")
exit()