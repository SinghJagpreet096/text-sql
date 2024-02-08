import pandas as pd 
from abc import ABC, abstractmethod
import logging
import numpy as np

from data_ingestion import load_data

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
    
class EmbeddingTransformer(AbstractDataTransformation):
    def __init__(self, X_train:pd.DataFrame, y_train:pd.DataFrame):
        self.X_train = X_train
        self.y_train = y_train
        # self.embedding_path = embedding_path

    def load_pretrained_embeddings(self):
        # Load pre-trained embeddings (GloVe in this example)
        word_vectors = api.load("glove-twitter-25")
        return word_vectors

    def create_embedding_matrix(self, word_index, word_vectors, embedding_dim):
        # Create an embedding matrix using pre-trained word vectors
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            if word in word_vectors:
                embedding_matrix[i] = word_vectors[word]
        return embedding_matrix

    def embeddings(self, embedding_dim=25):
        try:
            # Tokenize input sequences (questions)
            self.X_train['question'] = self.X_train["db_id"] + " " + self.X_train['question']
            self.X_train = self.X_train.drop('db_id', axis=1)
            tokenizer_input = Tokenizer()
            tokenizer_input.fit_on_texts(self.X_train['question'])
            X_train_sequences = tokenizer_input.texts_to_sequences(self.X_train['question'])
            X_train_padded = pad_sequences(X_train_sequences, padding='post')

            # Load pre-trained embeddings
            word_vectors = self.load_pretrained_embeddings()

            # Create embedding matrix
            embedding_matrix_input = self.create_embedding_matrix(tokenizer_input.word_index, word_vectors, embedding_dim)

            # Print the shape of the embedding matrix
            print(f"Embedding Matrix Shape (Input): {embedding_matrix_input.shape}")

            # Tokenize target sequences (SQL queries)
            tokenizer_output = Tokenizer()
            tokenizer_output.fit_on_texts(self.y_train['query'])
            y_train_sequences = tokenizer_output.texts_to_sequences(self.y_train['query'])
            y_train_padded = pad_sequences(y_train_sequences, padding='post', maxlen=X_train_padded.shape[1])

            # Create embedding matrix for output (assuming the same embedding matrix for both input and output)
            embedding_matrix_output = self.create_embedding_matrix(tokenizer_output.word_index, word_vectors, embedding_dim)

            # Print the shape of the embedding matrix
            print(f"Embedding Matrix Shape (Output): {embedding_matrix_output.shape}")
            print(f"X_train_padded type :{type(X_train_padded)}")
            print(f"y_train_padded type :{type(y_train_padded)}")
            print(f"embedding_matrix_input:{(embedding_matrix_input.shape)}")

            print(f"embedding_matrix_output:{(embedding_matrix_output.shape)}")

            return (
                embedding_matrix_input,
                embedding_matrix_output,
                X_train_padded,
                y_train_padded
            )
        except Exception as e:
            logging.error(f"Error in creating embeddings: {e}")
            raise e


# X_train, y_train = load_data("data/train/0000.parquet",["db_id","question","query"])
# embedding_matrix_input,embedding_matrix_output,X_train_padded,y_train_padded = EmbeddingTransformer(X_train,y_train).embeddings()

# print("embedding_matrix_input",type(embedding_matrix_input),embedding_matrix_input)
# print("embedding_matrix_input",type(embedding_matrix_output),embedding_matrix_output)
# print("X_train_padded",type(X_train_padded),X_train_padded)
# print("y_train_padded",type(y_train_padded),y_train_padded)


