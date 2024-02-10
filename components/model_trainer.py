from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from config import Config
import logging
from abc import ABC, abstractclassmethod


embedding_dim = 128
units = 64
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
    
    def model_compiler(self):
        pass

    def model_trainer(self):
        pass
            


