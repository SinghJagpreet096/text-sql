import re
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import Query, Request
from fastapi.responses import JSONResponse
from chainlit.logger import logger
from chainlit.config import config
from chainlit.server import app
import chainlit as cl
from fastapi.templating import Jinja2Templates
from db import generate_create_table_sql, seed_data, reset_db, execute_query

tokenizer = AutoTokenizer.from_pretrained("./model/")
model = AutoModelForCausalLM.from_pretrained("./model/")

conn = sqlite3.connect("pythonsqlite-v2.db")


@cl.oauth_callback
def auth_callback(provider_id: str, token: str, raw_user_data, default_app_user):
    if provider_id == "github":
        return default_app_user
    return None


@app.get("/project/translations")
async def project_translations(
    language: str = Query(default="en-US", description="Language code"),
):
    """Return project translations."""

    # Load translation based on the provided language
    translation = config.load_translation(language)

    return JSONResponse(
        content={
            "translation": translation,
        }
    )


# add a new post route to seed db with the queries provided by the user
@app.post("/seed")
async def seed(request: Request):
    """Seed the database with the queries provided by the user."""
    data = await request.json()
    seed_data_sql = data["seed_data_sql"]
    reset_db(conn)
    seed_data(conn, seed_data_sql)
    return JSONResponse(
        content={
            "message": "Data seeded successfully.",
        }
    )


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

    logger.info(f"Generated SQL query: {pred}")

    result = execute_query(conn=conn, query=pred)
    if len(result) > 0:
        result = result[0][0]
    else:
        result = "No results found."
    output = f"QUERY:{pred} \n\n result:{result}"

    await cl.Message(
        content=output,
    ).send()
