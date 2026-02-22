import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .db import get_conn
from .embeddings import embed_text
from .master_memory_extractor import extract_and_write_master_memory
from .summaries import maybe_update_summary

logger = logging.getLogger(__name__)
_ASYNC_WORKERS = max(1, int((os.getenv("CORTEX_LTM_ASYNC_WORKERS") or "2").strip() or "2"))
_SIDE_EFFECTS_EXECUTOR = ThreadPoolExecutor(max_workers=_ASYNC_WORKERS)


def _score_importance(actor: str, content: str) -> int:
    """
    v1 importance scoring for events.

    Scale:
      5 = identity/profile facts, explicit remember-intent
      3 = plans/commitments/constraints
      1 = mild preference/detail
      0 = chatter

    Only scores user events in v1.
    """
    if actor != "user":
        return 0

    t = (content or "").strip().lower()
    if not t:
        return 0

    # trivial chatter => 0
    trivial = {
        "ok",
        "okay",
        "k",
        "kk",
        "lol",
        "lmao",
        "bet",
        "nice",
        "cool",
        "word",
        "yup",
        "yeah",
        "nah",
        "thanks",
        "thx",
        "ty",
        "cya",
        "bye",
        "goodbye",
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "what's up",
        "whats up",
    }
    if t in trivial:
        return 0

    # 5 = identity/profile + explicit memory intent
    high = [
        "my name is",
        "call me ",
        "i am ",
        "i'm ",
        "my email",
        "my phone",
        "my address",
        "my birthday",
        "remember that",
        "remember this",
    ]
    if any(p in t for p in high):
        return 5

    # 3 = plans/constraints/commitments
    medium = [
        "i need to",
        "i want to",
        "we need to",
        "we should",
        "let's",
        "deadline",
        "plan",
        "goal is",
        "do not",
        "don't",
        "never ",
        "avoid ",
        "only ",
        "must ",
        "cannot ",
        "can't ",
    ]
    if any(p in t for p in medium):
        return 3

    # 1 = preferences / mild durable details
    low = [
        "i like",
        "i love",
        "i hate",
        "i prefer",
        "my favorite",
        "i don't like",
        "i dislike",
    ]
    if any(p in t for p in low):
        return 1

    # heuristic: longer user messages tend to contain signal
    if len(t) >= 120:
        return 1

    return 0


PROJECT_MEMORY_CUES = [
    ("project", "PROJECTS"),
    ("learning", "PROJECTS"),
    ("lesson", "PROJECTS"),
    ("plan", "PROJECTS"),
    ("vacation", "PROJECTS"),
    ("working on", "PROJECTS"),
    ("projects", "PROJECTS"),
    ("learn", "PROJECTS"),
    ("language", "PROJECTS"),
    ("coding", "PROJECTS"),
    ("memory layer", "LONG_RUNNING_CONTEXT"),
    ("memory specific", "LONG_RUNNING_CONTEXT"),
]


def _project_memory_bucket(content: str) -> str | None:
    t = (content or "").lower()
    for phrase, bucket in PROJECT_MEMORY_CUES:
        if phrase in t:
            return bucket
    return None


def _fetch_thread_user_id(thread_id: str) -> Optional[str]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select user_id from public.ltm_threads where id = %s limit 1;",
                (thread_id,),
            )
            row = cur.fetchone()
            return str(row[0]) if row and row[0] else None
    finally:
        conn.close()


def create_thread(user_id: str, title=None):
    """
    Creates a new thread in ltm_threads and returns the thread_id as a string.

    user_id: REQUIRED (uuid string) - identity key for cross-chat master memory
    """

    if not user_id or not str(user_id).strip():
        raise ValueError("create_thread: user_id is required")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.ltm_threads (user_id, title)
                values (%s, %s)
                returning id;
                """,
                (str(user_id), title),
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


def _submit_side_effect(task_name: str, fn, *args, **kwargs) -> None:
    future = _SIDE_EFFECTS_EXECUTOR.submit(fn, *args, **kwargs)

    def _on_done(done):
        try:
            done.result()
        except Exception:
            logger.exception("side-effect task failed: %s", task_name)

    future.add_done_callback(_on_done)


def _capture_master_memory_from_event(
    *,
    thread_id: str,
    event_id: str,
    content: str,
    importance_score: int,
    project_bucket: str | None,
) -> None:
    user_id = _fetch_thread_user_id(thread_id)
    if not user_id:
        return

    from .master_memory import upsert_master_item, add_master_evidence

    t = (content or "").strip()
    bucket = project_bucket or (
        "PROFILE"
        if any(p in t.lower() for p in ["my name is", "call me ", "i am ", "i'm "])
        else "LONG_RUNNING_CONTEXT"
    )

    stored_importance = max(importance_score, 5)
    master_id = upsert_master_item(
        user_id=user_id,
        bucket=bucket,
        text=t,
        confidence=0.70,
        stability="med",
        embed=False,
        meta={
            "source": "auto_event_capture",
            "importance_score": stored_importance,
            "capture_reason": "project_phrase" if project_bucket else "importance_high",
        },
    )

    add_master_evidence(
        master_item_id=master_id,
        thread_id=str(thread_id),
        event_id=str(event_id),
        weight=1.0,
        meta={"source": "auto_event_capture"},
    )


def _run_master_memory_extractor(thread_id: str) -> None:
    user_id = _fetch_thread_user_id(thread_id)
    if not user_id:
        return
    extract_and_write_master_memory(thread_id=thread_id, user_id=user_id)


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

    # v1 auto-rating + auto-embed:
    # - If caller leaves importance_score at 0, we score it.
    # - If importance_score >= 5, we force embed=True (early-memory buffer).
    if importance_score == 0:
        importance_score = _score_importance(actor, content)

    # Only embed raw events when we're very sure they're worth semantic recall.
    # (Keeps costs down and avoids indexing normal chatter.)
    if importance_score >= 5:
        embed = True

    emb = None
    if embed:
        try:
            emb = embed_text(content)
        except Exception as e:
            logger.warning(
                "event embedding failed; storing without embedding: %s: %s",
                type(e).__name__,
                e,
            )
            emb = None
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

        project_bucket = _project_memory_bucket(content)
        capture_forced = actor == "user" and (
            importance_score >= 5 or project_bucket is not None
        )
        if capture_forced:
            _submit_side_effect(
                "auto_event_capture",
                _capture_master_memory_from_event,
                thread_id=str(thread_id),
                event_id=str(event_id),
                content=content,
                importance_score=importance_score,
                project_bucket=project_bucket,
            )

        # Run extractor sparingly (batchy) to reduce cost + junk memories.
        # v1: only extract on very high-signal user turns.
        if actor == "user" and importance_score >= 5:
            _submit_side_effect("master_memory_extractor", _run_master_memory_extractor, str(thread_id))

        # Update summaries out of band so chat responses return faster.
        if actor == "assistant":
            _submit_side_effect("summary_update", maybe_update_summary, str(thread_id))

        return str(event_id)

    finally:
        conn.close()


def search_events_semantic(
    *,
    user_id: str,
    query: str,
    k: int = 5,
    thread_id: str | None = None,
    only_actor: str | None = "user",
    min_importance: int = 0,
):
    """
    Semantic search over ltm_events using pgvector distance (USER-SCOPED).

    - Embeds the query with OpenAI
    - Joins ltm_threads so we can filter by user_id (prevents cross-user bleed)
    - ORDER BY embedding <-> query_embedding
    - Optional thread_id filter
    - Optional actor filter (default: user)
    - Optional min_importance filter
    - Excludes rows where embedding IS NULL

    Returns list[dict] with:
      id, created_at, actor, content, meta, importance_score, distance
    """
    if not user_id or not str(user_id).strip():
        raise ValueError("search_events_semantic: user_id is required")

    q = (query or "").strip()
    if not q:
        raise ValueError("search_events_semantic: query is required")

    try:
        k_int = int(k)
    except Exception:
        raise ValueError("k must be an integer")

    if k_int < 1:
        k_int = 1
    if k_int > 50:
        k_int = 50  # cheap safety cap

    try:
        min_imp = int(min_importance)
    except Exception:
        min_imp = 0
    if min_imp < 0:
        min_imp = 0
    if min_imp > 10:
        min_imp = 10

    q_emb = embed_text(q)
    q_emb_literal = _vector_literal(q_emb)

    where = [
        "t.user_id = %s",
        "e.embedding is not null",
    ]
    params = [str(user_id), q_emb_literal]

    if thread_id:
        where.append("e.thread_id = %s")
        params.append(str(thread_id))

    if only_actor:
        where.append("e.actor = %s")
        params.append(str(only_actor))

    if min_imp > 0:
        where.append("e.importance_score >= %s")
        params.append(min_imp)

    # placeholders:
    # 1) vector for SELECT distance
    # 2) filters...
    # 3) vector for ORDER BY
    # 4) limit
    sql = f"""
        select
            e.id,
            e.created_at,
            e.actor,
            e.content,
            e.meta,
            e.importance_score,
            (e.embedding <-> (%s)::vector) as distance
        from public.ltm_events e
        join public.ltm_threads t on t.id = e.thread_id
        where {" and ".join(where)}
        order by e.embedding <-> (%s)::vector
        limit %s;
    """

    # Build params in correct order:
    # [vector_for_select] + filters(with vector already included) + [vector_for_order_by] + [limit]
    # Note: we already put q_emb_literal in params (for the WHERE list), but that is NOT used in SQL.
    # So we explicitly pass the vector twice: one for SELECT, one for ORDER BY.
    params_for_query = [q_emb_literal] + params[:-1] + [q_emb_literal, k_int]
    # Explanation:
    # - params = [user_id, q_emb_literal, ...filters]
    # - We don't actually need q_emb_literal inside filters; remove it by slicing params[:-1] is wrong if filters change.
    # Keep it simple instead:

    # Rebuild cleanly:
    filter_params = [str(user_id)]
    if thread_id:
        filter_params.append(str(thread_id))
    if only_actor:
        filter_params.append(str(only_actor))
    if min_imp > 0:
        filter_params.append(min_imp)

    params_for_query = [q_emb_literal] + filter_params + [q_emb_literal, k_int]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params_for_query))
            rows = cur.fetchall()

        out = []
        for id_, created_at, actor, content, meta, importance_score, distance in rows:
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
