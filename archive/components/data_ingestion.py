import pandas as pd 
import logging

class LoadData():

    def __init__(self,file_path):
        self.file_path = file_path

    def load_spider(self):
        logging.info("loading data")
        df = pd.read_parquet(self.file_path)
        X_train = df[["db_id","question"]]
        y_train = df[["query"]]
        logging.info(f"shape of X_train: {X_train.shape}")
        logging.info(f"shape of y_train: {y_train.shape}")
        X_train['question'] = X_train["db_id"] +" "+ X_train['question']
        X_train = X_train.drop('db_id',axis=1)
        return X_train,y_train

    def load_wikisql(self):

        train_df = pd.read_csv(self.file_path)
        train_df = train_df[train_df['sql'].apply(lambda x: len(x.split()) < 10)].reset_index()
        question, context, query = train_df['question'], train_df['table_header'], train_df['sql']
        return question, context, query


