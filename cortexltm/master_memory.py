# cortexltm/master_memory.py
"""
Master Memory (cross-chat) write-path.

This file intentionally starts "dumb":
- You can upsert memory items manually (no LLM extraction yet)
- You can attach evidence to items (thread/event/summary pointers)
- You can list top items for a user
- You can semantic-search master items (if embeddings exist)

Next step will add an LLM-based extractor that calls these primitives.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .db import get_conn
from .embeddings import embed_text

logger = logging.getLogger(__name__)

BUCKETS = {
    "PROFILE",
    "PREFERENCES",
    "CONSTRAINTS",
    "COMMUNICATION_STYLE",
    "LONG_RUNNING_CONTEXT",
    "GOALS",
    "PROJECTS",
    "NEXT_ACTIONS",
    "OPEN_LOOPS",
}

STATUSES = {"active", "deprecated", "conflicted"}
STABILITIES = {"high", "med", "low"}


def _vector_literal(vec: list[float] | None) -> str | None:
    if vec is None:
        return None
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _norm_bucket(bucket: str) -> str:
    b = (bucket or "").strip().upper()
    if b not in BUCKETS:
        raise ValueError(
            f"Invalid bucket '{bucket}'. Must be one of: {sorted(BUCKETS)}"
        )
    return b


def _norm_status(status: str) -> str:
    s = (status or "active").strip().lower()
    if s not in STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. Must be one of: {sorted(STATUSES)}"
        )
    return s


def _norm_stability(stability: str) -> str:
    st = (stability or "med").strip().lower()
    if st not in STABILITIES:
        raise ValueError(
            f"Invalid stability '{stability}'. Must be one of: {sorted(STABILITIES)}"
        )
    return st


def upsert_master_item(
    *,
    user_id: str,
    bucket: str,
    text: str,
    status: str = "active",
    stability: str = "med",
    confidence: float = 0.5,
    embed: bool = True,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Upsert a single master memory claim.

    v1 dedupe rule:
      (user_id, bucket, text) case-insensitive match with trim()

    Behavior:
      - If exists: update fields + increment reinforcement_count + timestamps
      - If not: insert new row

    Returns: master_item_id (uuid string)
    """
    if not user_id or not str(user_id).strip():
        raise ValueError("upsert_master_item: user_id is required")

    t = (text or "").strip()
    if not t:
        raise ValueError("upsert_master_item: text is required")

    b = _norm_bucket(bucket)
    s = _norm_status(status)
    st = _norm_stability(stability)

    try:
        conf = float(confidence)
    except Exception:
        raise ValueError("confidence must be a float")
    if conf < 0.0:
        conf = 0.0
    if conf > 1.0:
        conf = 1.0

    if meta is None:
        meta = {}

    emb_literal: str | None = None
    if embed:
        try:
            emb = embed_text(t)
            emb_literal = _vector_literal(emb)
        except Exception as e:
            # Do not fail writes if embedding provider is down.
            logger.warning(
                "master item embedding failed; storing without embedding: %s: %s",
                type(e).__name__,
                e,
            )
            emb_literal = None

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # find existing (case-insensitive, trim)
            cur.execute(
                """
                select id
                from public.ltm_master_items
                where user_id = %s
                  and bucket = %s
                  and lower(trim(text)) = lower(trim(%s))
                limit 1;
                """,
                (str(user_id), b, t),
            )
            row = cur.fetchone()

            if row:
                master_id = row[0]

                # NOTE: use COALESCE so a failed embedding doesn't clobber existing embedding.
                cur.execute(
                    """
                    update public.ltm_master_items
                    set
                      status = %s,
                      stability = %s,
                      confidence = %s,
                      reinforcement_count = reinforcement_count + 1,
                      last_seen_at = now(),
                      last_reinforced_at = now(),
                      meta = meta || (%s::jsonb),
                      embedding = coalesce((%s)::vector, embedding)
                    where id = %s;
                    """,
                    (
                        s,
                        st,
                        conf,
                        json.dumps(meta),
                        emb_literal,
                        master_id,
                    ),
                )
                conn.commit()
                return str(master_id)

            # insert new
            # NOTE: embeddings may be null; explicit cast is fine if the param is not null.
            cur.execute(
                """
                insert into public.ltm_master_items
                  (user_id, bucket, text, status, stability, confidence,
                   reinforcement_count, last_seen_at, last_reinforced_at, meta, embedding)
                values
                  (%s, %s, %s, %s, %s, %s,
                   1, now(), now(), %s::jsonb,
                   case when %s is null then null else (%s)::vector end)
                returning id;
                """,
                (
                    str(user_id),
                    b,
                    t,
                    s,
                    st,
                    conf,
                    json.dumps(meta),
                    emb_literal,
                    emb_literal,
                ),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
        return str(new_id)
    finally:
        conn.close()


def add_master_evidence(
    *,
    master_item_id: str,
    thread_id: Optional[str] = None,
    event_id: Optional[str] = None,
    summary_id: Optional[str] = None,
    weight: float = 1.0,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Attach evidence pointers to a master item (audit trail).

    At least one of thread_id/event_id/summary_id must be provided.
    """
    if not master_item_id or not str(master_item_id).strip():
        raise ValueError("add_master_evidence: master_item_id is required")

    if not (thread_id or event_id or summary_id):
        raise ValueError(
            "add_master_evidence: provide thread_id, event_id, and/or summary_id"
        )

    if meta is None:
        meta = {}

    try:
        w = float(weight)
    except Exception:
        w = 1.0
    if w <= 0:
        w = 1.0

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.ltm_master_evidence
                  (master_item_id, thread_id, event_id, summary_id, weight, meta)
                values
                  (%s, %s, %s, %s, %s, %s::jsonb)
                returning id;
                """,
                (
                    str(master_item_id),
                    thread_id,
                    event_id,
                    summary_id,
                    w,
                    json.dumps(meta),
                ),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
        return str(new_id)
    finally:
        conn.close()


def list_master_items(
    *,
    user_id: str,
    bucket: Optional[str] = None,
    limit: int = 50,
    status: str = "active",
) -> List[Dict[str, Any]]:
    """
    List master items for a user (default: active only), newest/reinforced first.
    """
    if not user_id or not str(user_id).strip():
        raise ValueError("list_master_items: user_id is required")

    try:
        lim = int(limit)
    except Exception:
        lim = 50
    if lim < 1:
        lim = 1
    if lim > 200:
        lim = 200

    where = ["user_id = %s"]
    params: List[Any] = [str(user_id)]

    if bucket:
        where.append("bucket = %s")
        params.append(_norm_bucket(bucket))

    if status:
        where.append("status = %s")
        params.append(_norm_status(status))

    params.append(lim)

    sql = f"""
        select
          id,
          created_at,
          updated_at,
          bucket,
          text,
          status,
          stability,
          confidence,
          reinforcement_count,
          last_seen_at,
          last_reinforced_at,
          meta
        from public.ltm_master_items
        where {" and ".join(where)}
        order by
          confidence desc,
          reinforcement_count desc,
          updated_at desc
        limit %s;
    """

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for (
            id_,
            created_at,
            updated_at,
            bucket_,
            text_,
            status_,
            stability_,
            confidence_,
            reinforcement_count,
            last_seen_at,
            last_reinforced_at,
            meta,
        ) in rows:
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {"_raw": meta}

            out.append(
                {
                    "id": str(id_),
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "bucket": bucket_,
                    "text": text_,
                    "status": status_,
                    "stability": stability_,
                    "confidence": float(confidence_),
                    "reinforcement_count": int(reinforcement_count),
                    "last_seen_at": last_seen_at,
                    "last_reinforced_at": last_reinforced_at,
                    "meta": meta,
                }
            )

        return out
    finally:
        conn.close()


def search_master_items_semantic(
    *,
    user_id: str,
    query: str,
    k: int = 8,
    bucket: Optional[str] = None,
    status: str = "active",
) -> List[Dict[str, Any]]:
    """
    Semantic search over ltm_master_items using pgvector distance.

    - Embeds the query with OpenAI
    - ORDER BY embedding <-> query_embedding
    - Requires items to have embeddings (embedding IS NOT NULL)
    - Optional bucket + status filters

    Returns list[dict] with:
      id, created_at, updated_at, bucket, text, status, stability, confidence,
      reinforcement_count, last_seen_at, last_reinforced_at, meta, distance
    """
    if not user_id or not str(user_id).strip():
        raise ValueError("search_master_items_semantic: user_id is required")

    q = (query or "").strip()
    if not q:
        raise ValueError("search_master_items_semantic: query is required")

    try:
        k_int = int(k)
    except Exception:
        k_int = 8
    if k_int < 1:
        k_int = 1
    if k_int > 50:
        k_int = 50

    q_emb = embed_text(q)
    q_emb_literal = _vector_literal(q_emb)

    where = ["user_id = %s", "embedding is not null"]
    filter_params: List[Any] = [str(user_id)]

    if bucket:
        where.append("bucket = %s")
        filter_params.append(_norm_bucket(bucket))

    if status:
        where.append("status = %s")
        filter_params.append(_norm_status(status))

    sql = f"""
        select
          id,
          created_at,
          updated_at,
          bucket,
          text,
          status,
          stability,
          confidence,
          reinforcement_count,
          last_seen_at,
          last_reinforced_at,
          meta,
          (embedding <-> (%s)::vector) as distance
        from public.ltm_master_items
        where {" and ".join(where)}
        order by embedding <-> (%s)::vector
        limit %s;
    """

    # Placeholder order in SQL is:
    # 1) distance vector, 2) filters (user_id, optional bucket/status), 3) order-by vector, 4) limit
    params_for_query: List[Any] = (
        [q_emb_literal] + filter_params + [q_emb_literal, k_int]
    )

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params_for_query))
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for (
            id_,
            created_at,
            updated_at,
            bucket_,
            text_,
            status_,
            stability_,
            confidence_,
            reinforcement_count,
            last_seen_at,
            last_reinforced_at,
            meta,
            distance,
        ) in rows:
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {"_raw": meta}

            out.append(
                {
                    "id": str(id_),
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "bucket": bucket_,
                    "text": text_,
                    "status": status_,
                    "stability": stability_,
                    "confidence": float(confidence_),
                    "reinforcement_count": int(reinforcement_count),
                    "last_seen_at": last_seen_at,
                    "last_reinforced_at": last_reinforced_at,
                    "meta": meta,
                    "distance": float(distance),
                }
            )

        return out
    finally:
        conn.close()
