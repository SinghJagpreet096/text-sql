import pandas as pd 
from abc import ABC, abstractmethod
import logging
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tiktoken

MAX_LEN = 20
enc = tiktoken.get_encoding('gpt2')
VOCAB_SIZE = enc.n_vocab
PADD_VALUE = 63

class AbstractDataTransformation(ABC):
    @abstractmethod
    def embeddings():
        pass
class Tokenizing(AbstractDataTransformation):
    def __init__(self, X_train:pd.DataFrame, y_train:pd.DataFrame):
        self.X_train = X_train
        self.y_train = y_train        
    def embeddings(self):
        try:
            # Tokenize input sequences (questions)
            tokenizer_input = Tokenizer()
            tokenizer_input.fit_on_texts(self.X_train['question'])
            X_train_sequences = tokenizer_input.texts_to_sequences(self.X_train['question'])
            X_train_padded = pad_sequences(X_train_sequences, padding='post')

            # print(f"X_train_padded: {X_train_padded[:10]}")

            # Tokenize target sequences (SQL queries)
            tokenizer_output = Tokenizer()
            tokenizer_output.fit_on_texts(self.y_train['query'])
            y_train_sequences = tokenizer_output.texts_to_sequences(self.y_train['query'])
            y_train_padded = pad_sequences(y_train_sequences, padding='post',maxlen=X_train_padded.shape[1])

            # print(f"y_train_padded: {y_train_padded[:10]}")

            # Vocabulary sizes
            vocab_size_input = len(tokenizer_input.word_index) + 1
            vocab_size_output = len(tokenizer_output.word_index) + 1

            print(f"vocab_size_input: {vocab_size_input}")
            print(f"vocab_size_output: {vocab_size_output}")

            return (
                vocab_size_input,
                vocab_size_output,
                X_train_padded,
                y_train_padded
                )
        except Exception as e:
            logging.error(f"error in tokenizing: {e}")

class TikTokenEncoding(AbstractDataTransformation):
    def __init__(self, question, context, query) -> None:
        self.question = question
        self.context = context
        self.query = query

    def preprocessing(self,column:pd.DataFrame,padd_val:int=-1):

        sequences = []
        for sent in column:
            print(enc.encode(sent))
            sequences.append(enc.encode(sent))
        padded = pad_sequences(sequences, padding='post',value=padd_val,maxlen=MAX_LEN)
        return padded
   
    def embeddings(self):
        padded_question = self.preprocessing(self.question,padd_val=PADD_VALUE)
        padded_context = self.preprocessing(self.context,padd_val=PADD_VALUE)
        padded_sql = self.preprocessing(self.query,padd_val=PADD_VALUE)
        return padded_question, padded_context, padded_sql, VOCAB_SIZE
        
