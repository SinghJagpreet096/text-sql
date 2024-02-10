import tiktoken
from abc import ABC,abstractclassmethod
import pandas as pd 

from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
from tensorflow.keras.models import load_model





enc = tiktoken.get_encoding("cl100k_base")

class PredictAbstract(ABC):

    @abstractclassmethod
    def inference_prediction():
        pass

    @abstractclassmethod
    def batch_prediction():
        pass


class PredictTiktoken(PredictAbstract):
    
    def __init__(self,model):
        self.model = model
    
    def inference_prediction(self,question:str, padding_index=0) -> str:
        question_sequence = enc.encode(question)
        # print(f"question_sequence: {question_sequence}")
        question_padded = pad_sequences([question_sequence], padding='post',maxlen=54)
        # print(f"question_padded: {question_padded}")
        predicted_sequence = self.model.predict(question_padded)
        # print(f"predicted_sequence: {predicted_sequence}")
        predicted_indices = np.argmax(predicted_sequence, axis=-1)
        predicted_indices[predicted_sequence.argmax(axis=-1) == padding_index] = padding_index
        predicted_query = enc.decode(predicted_indices[0])
        # predicted_sql_query = enc.decode(predicted_sequence)
        return predicted_query
    
    def batch_prediction(self, X_val: pd.DataFrame, y_val: pd.DataFrame):
        pred_queries = []
        true_queries = []
        for i in range(len(X_val)):
            question = X_val.loc[i, "question"]
            predicted_query = self.inference_prediction(question)
            pred_queries.append(predicted_query)
            true_queries.append(y_val.loc[i, 'query'])
            return true_queries, pred_queries
   

if __name__ == '__main__':
    model = load_model('/Users/jagpreetsingh/ML_Projects/text-sql/artifacts/tiktoken-enc.h5')
    # model = Predict(PredictTiktoken(model))
    pred = PredictTiktoken(model).inference_prediction('How many singers do we have?')
    print(pred) 