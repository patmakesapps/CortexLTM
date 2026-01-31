import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

db_url = os.getenv("SUPABASE_DB_URL")
if not db_url:
    raise RuntimeError("Missing SUPABASE_DB_URL in .env")


def main():
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute("select now();")
            row = cur.fetchone()
            print("DB time", row[0])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
