# create a sqlite database

import sqlite3
from sqlite3 import Error
from chainlit.logger import logger


def create_connection(db_file):
    """create a database connection to a SQLite database"""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


def create_table(conn, create_table_sql):
    """create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def seed_data(conn, seed_data_sql):
    """seed data from the seed_data_sql statement
    :param conn: Connection object
    :param seed_data_sql: a INSERT INTO statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(seed_data_sql)
        conn.commit()
    except Error as e:
        print(e)


# create a function to descibe all the table in the data base and return a concated string with a new line


def describe_table(conn):
    """describe all the tables in the database
    :param conn: Connection object
    :return:
    """
    try:
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = c.fetchall()
        for table in tables:
            c.execute(f"PRAGMA table_info({table[0]})")
            columns = c.fetchall()
            print(f"Table: {table[0]}")
            for column in columns:
                print(f"Column: {column[1]}")
    except Error as e:
        print(e)


def create_and_seed_tables(conn):
    create_table(
        conn,
        """CREATE TABLE stadium (
    stadium_id number,
    location text,
    name text,
    capacity number,
    highest number,
    lowest number,
    average number
    )""",
    )

    create_table(
        conn,
        """CREATE TABLE singer (
        singer_id number,
        name text,
        country text,
        song_name text,
        song_release_year text,
        age
        )""",
    )

    create_table(
        conn,
        """CREATE TABLE concert (
        concert_id number,
        concert_name text,
        theme text,
        stadium_id text,
        year text
        )""",
    )

    create_table(
        conn,
        """CREATE TABLE singer_in_concert (
        concert_id number,
        singer_id text
        )""",
    )
    seed_data(
        conn,
        """INSERT INTO stadium (stadium_id, location, name, capacity, highest, lowest, average) VALUES (1, 'Lagos', 'TBS', 10000, 12000, 8000, 10000)""",
    )
    seed_data(
        conn,
        """INSERT INTO singer (singer_id, name, country, song_name, song_release_year, age) VALUES (1, 'Davido', 'Nigeria', 'FEM', '2020', 28)""",
    )
    seed_data(
        conn,
        """INSERT INTO concert (concert_id, concert_name, theme, stadium_id, year) VALUES (1, 'FEM Concert', 'Afrobeat', '1', '2020')""",
    )
    seed_data(
        conn,
        """INSERT INTO singer_in_concert (concert_id, singer_id) VALUES (1, '1')""",
    )

    conn.commit()


def create_and_seed_table_v2(conn):
    # create database table for hospital management system
    create_table(
        conn,
        """CREATE TABLE hospital (
        hospital_id number,
        hospital_name text,
        location text,
        hospital_type text,
        hospital_capacity number
        )""",
    )

    create_table(
        conn,
        """CREATE TABLE patient (
        patient_id number,
        name text,
        age number
        )""",
    )

    create_table(
        conn,
        """CREATE TABLE doctor (
        doctor_id number,
        name text,
        specialization text
        )""",
    )

    create_table(
        conn,
        """CREATE TABLE appointment (
        appointment_id number,
        patient_id number,
        doctor_id number,
        hospital_id number,
        appointment_date text
        )""",
    )

    seed_data(
        conn,
        """INSERT INTO hospital (hospital_id, hospital_name, location, hospital_type, hospital_capacity) VALUES (1, 'LUTH', 'Lagos', 'Public', 1000)""",
    )

    # seed more data to patient table
    seed_data(
        conn,
        """INSERT INTO patient (patient_id, name, age) VALUES (1, 'John Doe', 30)""",
    )

    seed_data(
        conn,
        """INSERT INTO patient (patient_id, name, age) VALUES (2, 'John Doe', 20)""",
    )

    seed_data(
        conn,
        """INSERT INTO patient (patient_id, name, age) VALUES (3, 'John Doe', 40)""",
    )
    seed_data(
        conn,
        """INSERT INTO patient (patient_id, name, age) VALUES (4, 'John Doe', 25)""",
    )

    seed_data(
        conn,
        """INSERT INTO patient (patient_id, name, age) VALUES (5, 'John Doe', 45)""",
    )

    seed_data(
        conn,
        """INSERT INTO doctor (doctor_id, name, specialization) VALUES
        (1, 'Dr. John', 'Cardiologist')""",
    )

    seed_data(
        conn,
        """INSERT INTO appointment (appointment_id, patient_id, doctor_id, hospital_id, appointment_date) VALUES (1, 1, 1, 1, '2022-12-12')""",
    )

    conn.commit()


def get_create_table_cmd(table_name, cursor):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    column_defs = []
    for column in columns:
        column_name = column[1]
        data_type = column[2]
        not_null = "NOT NULL" if column[3] else ""
        default_value = f"DEFAULT {column[4]}" if column[4] is not None else ""
        pk = "PRIMARY KEY" if column[5] else ""
        column_def = (
            f"{column_name} {data_type} {not_null} {default_value} {pk}".strip()
        )
        column_defs.append(column_def)
    return f"CREATE TABLE {table_name} ({', '.join(column_defs)})"


def generate_create_table_sql(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    create_table_cmds = ""

    for table in tables:
        table_name = table[0]
        create_table_cmd = get_create_table_cmd(table_name, cursor)

        create_table_cmds += create_table_cmd + ";\n\n"

    return create_table_cmds


def execute_query(conn, query):
    try:
        cursor = conn.cursor()
        cursor.execute(query)
    except sqlite3.Error as e:
        logger.error(f"Error executing query: {e}")
        return []
    else:
        return cursor.fetchall()


def reset_db(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        table_name = table[0]
        cursor.execute(f"DROP TABLE {table_name}")
    conn.commit()
