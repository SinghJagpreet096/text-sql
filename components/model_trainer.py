from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Masking, Concatenate
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from config import Config

import logging
from abc import ABC, abstractclassmethod


# Model architecture
embedding_dim = 128
units = 128
MAX_LEN = Config.MAX_LEN
PADD_VAL = Config.PADD_VAL

class AbstractModelTrainer(ABC):
    @abstractclassmethod
    def model_compiler(self):
        pass

    @abstractclassmethod
    def model_trainer(self):
        pass
class BaseLineLSTM(AbstractModelTrainer):
    
    def model_compiler(self, vocab_size_input, vocab_size_output, X_train_padded):
        try:
            model = Sequential([
                Embedding(input_dim=vocab_size_input, output_dim=embedding_dim),
                LSTM(units, return_sequences=True),
                Dense(units=vocab_size_output, activation='softmax')
            ])

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            logging.info("model compiled")
            return model
        except Exception as e:
            logging.error(f"error in compiling model: {e}")
            raise e

    def model_trainer(self,model,X_train_padded,y_train_padded):
        try:

            history = model.fit(X_train_padded, y_train_padded, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, validation_split=Config.VALIDATION_SPLIT)
            return model, history
        except Exception as e:
            logging.error(f"error in training model: {e}")
            raise e
        
class AttentionLSTM(AbstractModelTrainer):
    def __init__(self,padded_question, padded_context, padded_sql,VOCAB_SIZE ):
        self.padded_question = padded_question
        self.padded_context = padded_context
        self.padded_sql = padded_sql
        self.vocab_size = VOCAB_SIZE
        
    def model_compiler(self):
        # Input layers
        input_question = Input(shape=(MAX_LEN,))
        input_context = Input(shape=(MAX_LEN,))

        ##masking
        input_question_masked = Masking(mask_value=PADD_VAL)(input_question)
        input_context_masked = Masking(mask_value=PADD_VAL)(input_question)

        # Embedding layers
        embedding_layer_question = Embedding(input_dim=self.vocab_size, output_dim=embedding_dim, )(input_question_masked)
        embedding_layer_context = Embedding(input_dim=self.vocab_size, output_dim=embedding_dim, )(input_context_masked)

        # LSTM layers
        lstm_layer_question = LSTM(units, return_sequences=True)(embedding_layer_question)
        lstm_layer_context = LSTM(units, return_sequences=True)(embedding_layer_context)

        # Attention layer
        attention = Attention()([lstm_layer_question, lstm_layer_context])

        # Concatenate the attention output with the LSTM output for context
        context_combined = Concatenate(axis=-1)([lstm_layer_context, attention])

        # Output layer
        output_layer = Dense(self.vocab_size, activation='softmax')(context_combined)

        # Model
        model = Model(inputs=[input_question, input_context], outputs=output_layer)
        model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

        return model


    def model_trainer(self,model):
        # Train the model
        history = model.fit([self.padded_question,self.padded_context], self.padded_sql, epochs=15, batch_size=64, validation_split=0.2)
        return history, model
        
            


