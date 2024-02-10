import sys
sys.path.append('/Users/jagpreetsingh/ML_Projects/text-sql/components')


from data_ingestion import LoadData
from data_tranformation import Tokenizing
from model_trainer import BaseLineLSTM
# from evaluate import Evaluate 
from config import Config

import logging

if __name__ == "__main__":

    if sys.argv[1] == '--BaseLineLSTM':

        ## load the data
        X_train, y_train = LoadData("data/train/0000.parquet").load_spider(Config.SELECTED_COLS)

        vocab_size_input,vocab_size_output,X_train_padded,y_train_padded = Tokenizing(X_train=X_train, y_train=y_train).embeddings()

        ## compile the model
        model = BaseLineLSTM().model_compiler(vocab_size_input=vocab_size_input, 
                                    vocab_size_output=vocab_size_output, 
                                    X_train_padded=X_train_padded)

        logging.info(f"model compiled")

        trained_model = BaseLineLSTM().model_trainer(model,X_train_padded,y_train_padded)

        logging.info(f"model trained")

        # Evaluate.evaluate()
    else:
        print("na")
    







