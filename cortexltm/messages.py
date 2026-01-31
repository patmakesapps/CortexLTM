from .db import get_conn


def create_thread(title=None):
    """
    Creates a new thread in ltm_threads
    and returns the thread_id as a string.
    """

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.ltm_threads (title)
                values (%s)
                returning id;
                """,
                (title,),
            )
            thread_id = cur.fetchone()[0]

        conn.commit()
        return str(thread_id)

    finally:
        conn.close()
