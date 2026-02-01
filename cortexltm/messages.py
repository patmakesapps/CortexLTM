import json
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


def add_event(thread_id, actor, content, meta=None, importance_score=0):
    """
    Adds one event (message) o ltm_events and returns the new event id.

    thread_id: the conversation UUID (string)
    actor: who produced it ("user", "assistant", etc)
    content: the text body
    meta: optional dict for extra info (source, modal, tags, etc)
    importance_score: small integer (0 normal, higher = more important)
    """

    if meta is None:
        meta = {}

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.ltm_events (thread_id, actor, content, meta, importance_score)
                values (%s, %s, %s, %s::jsonb, %s)
                returning id;
                """,
                (thread_id, actor, content, json.dumps(meta), importance_score),
            )
            event_id = cur.fetchone()[0]

        conn.commit()
        return str(event_id)

    finally:
        conn.close()
