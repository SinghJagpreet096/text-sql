# import requests
# import os


# API_TOKEN = os.getenv("HF_TOKEN")
# headers = {"Authorization": f"Bearer {API_TOKEN}"}
# API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b"
# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()
# data = query({"inputs": "The answer to the universe is"})
# print(data)
# !pip3 install -q -U bitsandbytes==0.42.0
# !pip3 install -q -U peft==0.8.2
# !pip3 install -q -U trl==0.7.10
# !pip3 install -q -U accelerate==0.27.1
# !pip3 install -q -U datasets==2.17.0
# !pip3 install -q -U transformers==4.38.0
from transformers import AutoModelForCausalLM,AutoTokenizer
import os

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")


tokenizer = AutoTokenizer.from_pretrained("singhjagpreet/gemma-2b_text_to_sql")
model_id = "singhjagpreet/gemma-2b_text_to_sql"
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map={"":0},
                                             token=os.environ['HF_TOKEN'])

text = """Question: What is the average number of working horses of farms with greater than 45 total number of horses?
Context: CREATE TABLE farm (Working_Horses INTEGER, Total_Horses INTEGER)"""
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
