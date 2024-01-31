from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from config import Config
import logging
from abc import ABC, abstractclassmethod






embedding_dim = 25
units = 256
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
                Embedding(input_dim=vocab_size_input[1], output_dim=embedding_dim, input_length=X_train_padded.shape[1]),
                LSTM(units, return_sequences=True),
                Dense(units=vocab_size_output[1], activation='softmax')
            ])

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            logging.info("model compiled")
            return model
        except Exception as e:
            logging.error(f"error in compiling model: {e}")

    def model_trainer(self,model,X_train_padded,y_train_padded):
        history = model.fit(X_train_padded.any(), y_train_padded.any(), epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, validation_split=Config.VALIDATION_SPLIT)
        return model, history


