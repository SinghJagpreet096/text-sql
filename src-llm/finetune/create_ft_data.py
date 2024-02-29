import pandas as pd
import sys



train = pd.read_parquet('data/train/train-00000-of-00001.parquet', columns=['question', 'query'])
test = pd.read_parquet('data/validation/validation-00000-of-00001.parquet', columns=['question', 'query'])

# print(data.head())


def create_finetune_data(data, filename):
    with open(filename, "w") as f:
        for i, row in data.iterrows():
            f.write(f"Question: {row['question']}\n")
            f.write(f"'<|startoftext|>{row['query']}'<|endoftext|>\n")
            if i > 100:
                break
## create training data
create_finetune_data(train, "spider_train.txt")

# create test data  
create_finetune_data(test, "spider_test.txt")


