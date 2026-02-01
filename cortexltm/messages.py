import json
from .db import get_conn
from .embeddings import embed_text
from .summaries import maybe_update_summary


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


def _vector_literal(vec: list[float] | None) -> str | None:
    """
    Convert a Python list of floats into pgvector text format: '[1,2,3]'
    Returns None for NULL.
    """
    if vec is None:
        return None
    # Use repr/str for floats; keep it simple.
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def add_event(
    thread_id, actor, content, meta=None, importance_score=0, embed: bool = False
):
    """
    Adds one event (message) to ltm_events and returns the new event id.

    thread_id: the conversation UUID (string)
    actor: who produced it ("user", "assistant", etc)
    content: the text body
    meta: optional dict for extra info (source, modal, tags, etc)
    importance_score: small integer (0 normal, higher = more important)
    embed: if True, stores an OpenAI embedding in ltm_events.embedding
    """

    if meta is None:
        meta = {}

    emb = embed_text(content) if embed else None
    emb_literal = _vector_literal(emb)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.ltm_events (thread_id, actor, content, meta, importance_score, embedding)
                values (%s, %s, %s, %s::jsonb, %s, (%s)::vector)
                returning id;
                """,
                (
                    thread_id,
                    actor,
                    content,
                    json.dumps(meta),
                    importance_score,
                    emb_literal,
                ),
            )
            event_id = cur.fetchone()[0]

        conn.commit()

        # v1: update summaries after a full turn completes (assistant written)
        if actor == "assistant":
            try:
                maybe_update_summary(str(thread_id))
            except Exception:
                pass

        return str(event_id)

    finally:
        conn.close()


def search_events_semantic(query: str, k: int = 5, thread_id: str | None = None):
    """
    Semantic search over ltm_events using pgvector distance.

    - Embeds the query with OpenAI
    - ORDER BY embedding <-> query_embedding
    - Optional thread_id filter
    - Excludes rows where embedding IS NULL
    - Returns list[dict] with:
        id, created_at, actor, content, meta, importance_score, distance
    """
    if k is None:
        k = 5
    try:
        k_int = int(k)
    except Exception:
        raise ValueError("k must be an integer")

    if k_int < 1:
        k_int = 1
    if k_int > 50:
        k_int = 50  # cheap safety cap

    q_emb = embed_text(query)
    q_emb_literal = _vector_literal(q_emb)

    where = ["embedding is not null"]
    params = [q_emb_literal]

    if thread_id:
        where.append("thread_id = %s")
        params.append(thread_id)

    params.append(k_int)

    sql = f"""
        select
            id,
            created_at,
            actor,
            content,
            meta,
            importance_score,
            (embedding <-> (%s)::vector) as distance
        from public.ltm_events
        where {" and ".join(where)}
        order by embedding <-> (%s)::vector
        limit %s;
    """

    # Note: we need the query vector twice because we use it in SELECT + ORDER BY.
    # Keep it explicit and simple.
    params_for_query = [params[0]] + [params[0]] + params[1:]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params_for_query))
            rows = cur.fetchall()

        out = []
        for id_, created_at, actor, content, meta, importance_score, distance in rows:
            # psycopg2 may return jsonb as dict or as str depending on environment
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {"_raw": meta}

            out.append(
                {
                    "id": str(id_),
                    "created_at": created_at,
                    "actor": actor,
                    "content": content,
                    "meta": meta,
                    "importance_score": int(importance_score),
                    "distance": float(distance),
                }
            )

        return out

    finally:
        conn.close()
