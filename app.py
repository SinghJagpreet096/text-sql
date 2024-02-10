import chainlit as cl
from keras.models import load_model
from pipelines.predict_pipeline import PredictTiktoken

model = load_model('/Users/jagpreetsingh/ML_Projects/text-sql/artifacts/masking-tiktoken.h5')
@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...

    pred = PredictTiktoken(model).inference_prediction(message.content)
    print(pred)




    # Send a response back to the user
    await cl.Message(
        content=pred,
    ).send()
