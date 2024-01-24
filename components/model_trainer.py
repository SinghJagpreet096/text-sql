from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from config import Config


embedding_dim = 128
units = 256

class Model_Trainer:
    
    def model_compiler(vocab_size_input, vocab_size_output, X_train_padded):
        model = Sequential([
            Embedding(input_dim=vocab_size_input, output_dim=embedding_dim, input_length=X_train_padded.shape[1]),
            LSTM(units, return_sequences=True),
            Dense(units=vocab_size_output, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


        return model

    def model_trainer(model,X_train_padded,y_train_padded):
        history = model.fit(X_train_padded, y_train_padded, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, validation_split=Config.VALIDATION_SPLIT)
        return model, history

