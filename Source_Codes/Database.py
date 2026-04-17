import pandas as pd
import psycopg2
from sqlalchemy import create_engine

def load_data():
    conn = psycopg2.connect(
        dbname="healthcare_db",
        user="postgres",
        password="Rahym2008!", # Correct password
        host="localhost",
        port="5432"
    )
    query = "SELECT * FROM healthcare.healthcare_raw;"
    healthcare_data = pd.read_sql(query, conn)
    conn.close()
    return healthcare_data

def load_clean_data_to_db(healthcare_data):
    # USE THE REAL PASSWORD HERE TOO
    engine = create_engine(
        "postgresql+psycopg2://postgres:Rahym2008!@localhost:5432/healthcare_db"
    )
    healthcare_data.to_sql(
        name="healthcare_cleaned",
        con=engine,
        schema="healthcare",
        if_exists="replace",
        index=False
    )
    print("\n Cleaned data loaded into PostgreSQL")