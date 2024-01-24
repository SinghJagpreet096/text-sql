import pandas as pd 
import logging


def load_data(file_path:str, selected_columns:list):
    logging.info("loading data")
    df = pd.read_parquet(file_path)
    X_train = df[["db_id","question"]]
    y_train = df[["query"]]
    logging.info(f"shape of X_train: {X_train.shape}")
    logging.info(f"shape of y_train: {y_train.shape}")
    return X_train,y_train


