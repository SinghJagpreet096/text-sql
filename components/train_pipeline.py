from data_ingestion import load_data
from data_tranformation import Data_Transformation
from model_trainer import Model_Trainer

import logging







if __name__ == "__main__":

    ## load the data
    X_train, y_train = load_data("data/train/0000.parquet",["db_id","question","query"])


    ## transform the data
    vocab_size_input,vocab_size_output,X_train_padded,y_train_padded = Data_Transformation(X_train=X_train, y_train=y_train).embeddings()

    ## compile the model

    model = Model_Trainer.model_compiler(vocab_size_input=vocab_size_input, 
                                 vocab_size_output=vocab_size_output, 
                                 X_train_padded=X_train_padded)

    logging.info(f"model compiled")

    trained_model = Model_Trainer.model_trainer(model,X_train_padded,y_train_padded)

    logging.info(f"model trained")







