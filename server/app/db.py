import psycopg2
from config.settings import PG_DATABASE, PG_USER, PG_PASSWORD, PG_HOST,PG_PORT

def get_connection():
    conn = psycopg2.connect(
        dbname=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
        PG_HOST=PG_HOST,
        port=PG_PORT
    )
    return conn