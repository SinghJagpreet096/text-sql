import sys
sys.path.append('../text-sql/')

from abc import ABC, abstractclassmethod
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from  pipelines.predict_pipeline import PredictTiktoken
import pandas as pd
from data_ingestion import LoadData
from config import Config
from keras.models import load_model

class EvaluateAbstract(ABC):
    @abstractclassmethod
    def evaluate():
        pass

class BlueScore(EvaluateAbstract):
    def __init__(self,model):
        self.model = model

    def evaluate(self, X_val: pd.DataFrame, y_val: pd.DataFrame):
        smoothing_function = SmoothingFunction().method1
        y_test, decoded_predictions = PredictTiktoken(self.model).batch_prediction(X_val, y_val)
        bleu_scores = [sentence_bleu([true_query.split()], generated_query.split(),smoothing_function=smoothing_function) for true_query, generated_query in zip(y_test, decoded_predictions)]
        return bleu_scores


        
if __name__== "__main__":
    X_val, y_val = LoadData('data/validation/validation-00000-of-00001.parquet').load_spider()
    model = load_model('artifacts/masking-tiktoken.h5')
    score = BlueScore(model).evaluate(X_val,y_val)
    average_score = sum(score) /len(score)
    print(average_score)
