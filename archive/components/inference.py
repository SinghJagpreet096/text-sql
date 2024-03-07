from tensorflow.keras.models import load_model


model = load_model("artifacts/baseline.h5")


def inference(question):
    pass