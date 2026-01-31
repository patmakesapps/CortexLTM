import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()


def get_conn():
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("Missing SUPABASE_DB_URL in .env")
    return psycopg2.connect(db_url)
