from transformers import AutoTokenizer, AutoModelForCausalLM
import chainlit as cl
import re
from langchain.prompts import ChatPromptTemplate
from db_seedr import generate_create_table_sql, execute_query
import sqlite3
from sqlite3 import Error

tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-350M")
model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-350M")

conn = sqlite3.connect("pythonsqlite-v2.db")


# PROMT = """CREATE TABLE stadium (
#     stadium_id number,
#     location text,
#     name text,
#     capacity number,
#     highest number,
#     lowest number,
#     average number
# )

# CREATE TABLE singer (
#     singer_id number,
#     name text,
#     country text,
#     song_name text,
#     song_release_year text,
#     age number,
#     is_male others
# )

# CREATE TABLE concert (
#     concert_id number,
#     concert_name text,
#     theme text,
#     stadium_id text,
#     year text
# )

# CREATE TABLE singer_in_concert (
#     concert_id number,
#     singer_id text
# )

# -- Using valid SQLite, answer the following questions for the tables provided above.

# -- What is the maximum, the average, and the minimum capacity of stadiums ?

# SELECT"""


# chat_template = ChatPromptTemplate.from_messages([
#     ("human", "What is the capital of {country}?"),
#     ("ai", "The capital of {country} is {capital}.")
# ])


# messages = chat_template.format_messages(country="Canada", capital="Ottawa")
# print(messages)
# @cl.password_auth_callback
# def auth_callback(username: str, password: str):
#     # Fetch the user matching username from your database
#     # and compare the hashed password with the value stored in the database
#     if (username, password) == ("admin", "admin"):
#         return cl.User(
#             identifier="admin", metadata={"role": "admin", "provider": "credentials"}
#         )
#     else:
#         return None


# @cl.on_chat_start
# async def on_chat_start():
#     app_user = cl.user_session.get("user")
#     await cl.Message(f"Hello {app_user.identifier}").send()


@cl.oauth_callback
def auth_callback(provider_id: str, token: str, raw_user_data, default_app_user):
    if provider_id == "github":
        return default_app_user
    return None


@cl.on_message
async def main(message: cl.Message):
    query = message.content
    create_sql = generate_create_table_sql(conn)
    # print(create_sql)
    PROMT = (
        create_sql
        + f"""

    -- Using valid SQLite, answer the following questions for the tables provided above.

    -- {query}

    SELECT"""
    )

    input_ids = tokenizer(PROMT, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=500)
    pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Send a response back to the user
    if "SELECT" in pred:
        pred = pred.split("SELECT")[1]
    pred = "SELECT " + re.sub('"""', "", pred)

    result = execute_query(conn=conn, query=pred)

    if len(result) > 0:
        result = result[0][0]

    output = f"QUERY:{pred} \n\n result:{result}"

    await cl.Message(
        content=output,
    ).send()
