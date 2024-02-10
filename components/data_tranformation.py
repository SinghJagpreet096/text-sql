import pandas as pd 
from abc import ABC, abstractmethod
import logging
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# >>>
# >>> model = api.load("glove-twitter-25")  # load glove vectors
# >>> model.most_similar("cat")  # show words that similar to word 'cat'


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



