import os
import re
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from .db import get_conn
from .messages import add_event, create_thread

SUMMARY_CUE_REGEX = re.compile(
    r"\b(recap|summari[sz]e|catch me up|where were we|continue)\b", re.IGNORECASE
)
SEMANTIC_CUE_REGEX = re.compile(
    r"\b(remember|what did i say|what was the plan|who am i|my name)\b",
    re.IGNORECASE,
)

app = FastAPI(title="CortexLTM API", version="0.1.0")


class ThreadCreateRequest(BaseModel):
    user_id: str
    title: str | None = None


class EventCreateRequest(BaseModel):
    actor: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=6000)
    meta: dict[str, Any] | None = None


class MemoryContextRequest(BaseModel):
    latest_user_text: str
    short_term_limit: int | None = Field(default=30, ge=1, le=200)


def _validate_api_key(x_api_key: str | None) -> None:
    expected = os.getenv("CORTEXLTM_API_KEY")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _normalize_limit(raw_limit: int | None, default: int, cap: int) -> int:
    if raw_limit is None:
        return default
    limit = int(raw_limit)
    if limit < 1:
        return 1
    if limit > cap:
        return cap
    return limit


def _to_iso(value: datetime | None) -> str | None:
    if not value:
        return None
    return value.isoformat()


def _query_threads(user_id: str, limit: int) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select id, user_id, title, created_at
                from public.ltm_threads
                where user_id = %s
                order by created_at desc
                limit %s;
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "id": str(id_),
                "user_id": str(user_id_),
                "title": title,
                "created_at": _to_iso(created_at),
            }
            for id_, user_id_, title, created_at in rows
        ]
    finally:
        conn.close()


def _query_events(thread_id: str, limit: int) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select id, thread_id, actor, content, meta, created_at
                from public.ltm_events
                where thread_id = %s
                order by created_at desc
                limit %s;
                """,
                (thread_id, limit),
            )
            rows = cur.fetchall()
        rows.reverse()
        out: list[dict[str, Any]] = []
        for id_, thread_id_, actor, content, meta, created_at in rows:
            if actor not in ("user", "assistant"):
                continue
            out.append(
                {
                    "id": str(id_),
                    "thread_id": str(thread_id_),
                    "role": actor,
                    "content": content,
                    "meta": meta or {},
                    "created_at": _to_iso(created_at),
                }
            )
        return out
    finally:
        conn.close()


def _get_active_summary(thread_id: str) -> str | None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select summary
                from public.ltm_thread_summaries
                where thread_id = %s and is_active = true
                order by created_at desc
                limit 1;
                """,
                (thread_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        summary = row[0]
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
        return None
    finally:
        conn.close()


def _get_semantic_memories(thread_id: str, limit: int) -> list[str]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select text
                from public.ltm_master_items
                where user_id = (select user_id from public.ltm_threads where id = %s)
                  and status = 'active'
                order by updated_at desc
                limit %s;
                """,
                (thread_id, limit),
            )
            rows = cur.fetchall()
        memories: list[str] = []
        for (value,) in rows:
            if isinstance(value, str) and value.strip():
                memories.append(value.strip())
        return memories
    finally:
        conn.close()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/threads")
def create_thread_route(
    payload: ThreadCreateRequest, x_api_key: str | None = Header(default=None)
) -> dict[str, str]:
    _validate_api_key(x_api_key)
    thread_id = create_thread(payload.user_id, payload.title)
    return {"thread_id": thread_id}


@app.get("/v1/threads")
def list_threads_route(
    user_id: str = Query(...),
    limit: int = Query(50),
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _validate_api_key(x_api_key)
    rows = _query_threads(user_id=user_id, limit=_normalize_limit(limit, 50, 200))
    return {"threads": rows}


@app.get("/v1/threads/{thread_id}/events")
def list_events_route(
    thread_id: str,
    limit: int = Query(100),
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _validate_api_key(x_api_key)
    events = _query_events(thread_id=thread_id, limit=_normalize_limit(limit, 100, 200))
    return {"thread_id": thread_id, "messages": events}


@app.post("/v1/threads/{thread_id}/events")
def create_event_route(
    thread_id: str,
    payload: EventCreateRequest,
    x_api_key: str | None = Header(default=None),
) -> dict[str, str]:
    _validate_api_key(x_api_key)
    event_id = add_event(
        thread_id=thread_id,
        actor=payload.actor,
        content=payload.content,
        meta=payload.meta or {},
    )
    return {"event_id": event_id}


@app.get("/v1/threads/{thread_id}/summary")
def get_summary_route(
    thread_id: str, x_api_key: str | None = Header(default=None)
) -> dict[str, Any]:
    _validate_api_key(x_api_key)
    return {"thread_id": thread_id, "summary": _get_active_summary(thread_id)}


@app.post("/v1/threads/{thread_id}/memory-context")
def build_memory_context_route(
    thread_id: str,
    payload: MemoryContextRequest,
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _validate_api_key(x_api_key)
    context: list[dict[str, str]] = []

    if SUMMARY_CUE_REGEX.search(payload.latest_user_text):
        summary = _get_active_summary(thread_id)
        if summary:
            context.append({"role": "system", "content": f"Active summary:\n{summary}"})

    if SEMANTIC_CUE_REGEX.search(payload.latest_user_text):
        semantic = _get_semantic_memories(thread_id, 5)
        if semantic:
            context.append(
                {
                    "role": "system",
                    "content": "Relevant long-term memory:\n" + "\n- ".join(semantic),
                }
            )

    recent = _query_events(
        thread_id=thread_id,
        limit=_normalize_limit(payload.short_term_limit, 30, 200),
    )
    for message in recent:
        context.append({"role": message["role"], "content": message["content"]})

    return {"thread_id": thread_id, "messages": context}
