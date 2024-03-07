from datasets import list_datasets
from tqdm import tqdm
from datasets import list_datasets, load_dataset
import pandas as pd
import os
import sys


os.makedirs('data',exist_ok=True)
os.makedirs('data/train',exist_ok=True)
os.makedirs('data/validation',exist_ok=True)
os.makedirs('data/test',exist_ok=True)
## load dataset
dataset = load_dataset('wikisql')

## iterate over train set and create question/sql pairs
train_set = []
for row in tqdm(dataset['train']):
    # print(row)
    # break
    data = {
        'question': row['question'],
        'sql': row['sql']['human_readable'],
        'table_header': row['table']['header'],
        'table_header_types': row['table']['types'],
        'conds': row['sql']['conds']['condition'],
        'rows': row['table']['rows']
            }

    train_set.append(data)

## iterate over validation set and create question/sql pairs
validation_set = []
for row in tqdm(dataset['validation']):
    data = {
        'question': row['question'],
        'sql': row['sql']['human_readable'],
        'table_header': row['table']['header'],
        'table_header_types': row['table']['types'],
        'conds': row['sql']['conds']['condition'],
        'rows': row['table']['rows']
            }
    validation_set.append(data)
## iterate over test set and create question/sql pairs
test_set = []
for row in tqdm(dataset['test']):
    data = {
        'question': row['question'],
        'sql': row['sql']['human_readable'],
        'table_header': row['table']['header'],
        'table_header_types': row['table']['types'],
        'conds': row['sql']['conds']['condition'],
        'rows': row['table']['rows']
            }
    test_set.append(data)


train_df, validation_df, test_df = pd.DataFrame(train_set), pd.DataFrame(validation_set), pd.DataFrame(test_set)

train_df.to_csv('data/train/wikisql_train.csv', index=False)
validation_df.to_csv('data/validation/wikisql_val.csv', index=False)
test_df.to_csv('data/test/wikisql_test.csv', index=False)