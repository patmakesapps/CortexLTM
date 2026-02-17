import os
import re
import json
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .db import DatabaseUnavailableError, get_conn
from .llm import chat_reply
from .messages import add_event, create_thread
from .summaries import force_update_summary

SUMMARY_CUE_REGEX = re.compile(
    r"\b(recap|summari[sz]e|catch me up|where were we|continue)\b", re.IGNORECASE
)
SEMANTIC_CUE_REGEX = re.compile(
    r"\b(remember|what did i say|what was the plan|who am i|my name)\b",
    re.IGNORECASE,
)

app = FastAPI(title="CortexLTM API", version="0.1.0")


@app.exception_handler(DatabaseUnavailableError)
def handle_db_unavailable(
    _request: Request, exc: DatabaseUnavailableError
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


class ThreadCreateRequest(BaseModel):
    user_id: str | None = None
    title: str | None = None


class ThreadRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class EventCreateRequest(BaseModel):
    actor: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=6000)
    meta: dict[str, Any] | None = None


class EventReactionRequest(BaseModel):
    reaction: str | None = Field(
        default=None, pattern="^(thumbs_up|heart|angry|sad|brain)$"
    )


class MemoryContextRequest(BaseModel):
    latest_user_text: str
    short_term_limit: int | None = Field(default=30, ge=1, le=200)


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    short_term_limit: int | None = Field(default=30, ge=1, le=200)


def _validate_api_key(x_api_key: str | None) -> None:
    expected = os.getenv("CORTEXLTM_API_KEY")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    raw = authorization.strip()
    if not raw:
        return None
    if raw.lower().startswith("bearer "):
        token = raw[7:].strip()
        return token or None
    return None


def _fetch_supabase_user_id(access_token: str) -> str:
    supabase_url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    supabase_anon_key = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
    if not supabase_url or not supabase_anon_key:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL and SUPABASE_ANON_KEY are required in AUTH_MODE=supabase.",
        )

    req = urllib.request.Request(
        f"{supabase_url}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {access_token}",
            "apikey": supabase_anon_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            body = res.read().decode("utf-8")
    except urllib.error.HTTPError:
        raise HTTPException(status_code=401, detail="Invalid or expired access token.")
    except Exception:
        raise HTTPException(status_code=503, detail="Auth provider unavailable.")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=503, detail="Auth provider returned invalid JSON.")

    user_id = payload.get("id")
    if not isinstance(user_id, str) or not user_id.strip():
        raise HTTPException(status_code=401, detail="Access token missing user id.")
    return user_id.strip()


def _authorize_request(
    x_api_key: str | None, authorization: str | None
) -> str | None:
    _validate_api_key(x_api_key)
    auth_mode = (os.getenv("AUTH_MODE") or "dev").strip().lower()
    if auth_mode != "supabase":
        return None

    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Bearer token required.")
    return _fetch_supabase_user_id(token)


def _resolve_effective_user_id(requested_user_id: str | None, auth_user_id: str | None) -> str:
    if auth_user_id:
        return auth_user_id
    if not requested_user_id or not requested_user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required.")
    return requested_user_id.strip()


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
                select id, user_id, title, created_at, meta
                from public.ltm_threads
                where user_id = %s
                order by created_at desc
                limit %s;
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()
        return [
            _thread_row_to_payload(id_, user_id_, title, created_at, meta)
            for id_, user_id_, title, created_at, meta in rows
        ]
    finally:
        conn.close()


def _thread_row_to_payload(
    id_: Any, user_id_: Any, title: Any, created_at: Any, meta: Any
) -> dict[str, Any]:
    data = meta if isinstance(meta, dict) else {}
    is_core_memory = bool(data.get("is_core_memory"))
    return {
        "id": str(id_),
        "user_id": str(user_id_),
        "title": title,
        "created_at": _to_iso(created_at),
        "is_core_memory": is_core_memory,
    }


def _query_events(
    thread_id: str, limit: int, reaction_user_id: str | None = None
) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if not reaction_user_id:
                cur.execute(
                    """
                    select id, thread_id, actor, content, meta, created_at, null::text as reaction
                    from public.ltm_events
                    where thread_id = %s
                    order by created_at desc
                    limit %s;
                    """,
                    (thread_id, limit),
                )
            else:
                try:
                    cur.execute(
                        """
                        select
                          e.id,
                          e.thread_id,
                          e.actor,
                          e.content,
                          e.meta,
                          e.created_at,
                          r.reaction
                        from public.ltm_events e
                        left join public.ltm_event_reactions r
                          on r.event_id = e.id
                         and r.user_id = %s
                        where e.thread_id = %s
                        order by e.created_at desc
                        limit %s;
                        """,
                        (reaction_user_id, thread_id, limit),
                    )
                except Exception:
                    # Keep message reads working before reaction migration is applied.
                    cur.execute(
                        """
                        select id, thread_id, actor, content, meta, created_at, null::text as reaction
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
        for id_, thread_id_, actor, content, meta, created_at, reaction in rows:
            if actor not in ("user", "assistant"):
                continue
            merged_meta = meta if isinstance(meta, dict) else {}
            if actor == "assistant" and isinstance(reaction, str) and reaction.strip():
                merged_meta = {**merged_meta, "reaction": reaction.strip()}
            out.append(
                {
                    "id": str(id_),
                    "thread_id": str(thread_id_),
                    "role": actor,
                    "content": content,
                    "meta": merged_meta,
                    "created_at": _to_iso(created_at),
                }
            )
        return out
    finally:
        conn.close()


def _assert_thread_owner(thread_id: str, auth_user_id: str | None) -> None:
    if not auth_user_id:
        return
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select 1
                from public.ltm_threads
                where id = %s and user_id = %s
                limit 1;
                """,
                (thread_id, auth_user_id),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Thread not found.")
    finally:
        conn.close()


def _get_thread_user_id(thread_id: str) -> str | None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select user_id
                from public.ltm_threads
                where id = %s
                limit 1;
                """,
                (thread_id,),
            )
            row = cur.fetchone()
        if not row or not row[0]:
            return None
        return str(row[0])
    finally:
        conn.close()


def _resolve_reaction_user_id(thread_id: str, auth_user_id: str | None) -> str | None:
    if auth_user_id:
        return auth_user_id
    return _get_thread_user_id(thread_id)


def _event_exists_and_actor(thread_id: str, event_id: str) -> str | None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select actor
                from public.ltm_events
                where id = %s and thread_id = %s
                limit 1;
                """,
                (event_id, thread_id),
            )
            row = cur.fetchone()
        if not row:
            return None
        return str(row[0])
    finally:
        conn.close()


def _normalize_reaction_event_id(event_id: str) -> str:
    raw = (event_id or "").strip()
    if raw.startswith("assistant-"):
        raw = raw[len("assistant-") :].strip()
    try:
        return str(UUID(raw))
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Event not found.") from exc


def _set_event_reaction(
    thread_id: str,
    event_id: str,
    user_id: str,
    reaction: str | None,
) -> tuple[str | None, bool]:
    actor = _event_exists_and_actor(thread_id, event_id)
    if actor is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    if actor != "assistant":
        raise HTTPException(status_code=400, detail="Reactions can only target assistant events.")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if reaction is None:
                cur.execute(
                    """
                    delete from public.ltm_event_reactions
                    where event_id = %s and user_id = %s;
                    """,
                    (event_id, user_id),
                )
                conn.commit()
                return None, False

            cur.execute(
                """
                insert into public.ltm_event_reactions (event_id, user_id, reaction, meta)
                values (%s, %s, %s, '{}'::jsonb)
                on conflict (event_id, user_id)
                do update set
                  reaction = excluded.reaction,
                  updated_at = now()
                returning reaction;
                """,
                (event_id, user_id, reaction),
            )
            stored = cur.fetchone()
            conn.commit()
            stored_reaction = str(stored[0]) if stored and stored[0] else reaction
            summary_updated = False
            if stored_reaction == "brain":
                summary_updated = force_update_summary(thread_id)
            return stored_reaction, summary_updated
    finally:
        conn.close()


def _rename_thread(thread_id: str, title: str) -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.ltm_threads
                set title = %s
                where id = %s;
                """,
                (title, thread_id),
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Thread not found.")
            conn.commit()
    finally:
        conn.close()


def _delete_thread(thread_id: str, auth_user_id: str | None) -> bool:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                delete from public.ltm_master_evidence e
                where
                  e.thread_id = %s
                  or e.event_id in (
                    select ev.id
                    from public.ltm_events ev
                    where ev.thread_id = %s
                  )
                  or e.summary_id in (
                    select s.id
                    from public.ltm_thread_summaries s
                    where s.thread_id = %s
                  );
                """,
                (thread_id, thread_id, thread_id),
            )
            if auth_user_id:
                cur.execute(
                    """
                    delete from public.ltm_threads
                    where id = %s and user_id = %s;
                    """,
                    (thread_id, auth_user_id),
                )
                if cur.rowcount > 0:
                    conn.commit()
                    return True
                cur.execute(
                    """
                    select 1
                    from public.ltm_threads
                    where id = %s
                    limit 1;
                    """,
                    (thread_id,),
                )
                if cur.fetchone():
                    raise HTTPException(status_code=404, detail="Thread not found.")
                return False

            cur.execute(
                """
                delete from public.ltm_threads
                where id = %s;
                """,
                (thread_id,),
            )
            deleted = cur.rowcount > 0
            if deleted:
                conn.commit()
            return deleted
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


def _mark_thread_core_memory(thread_id: str) -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.ltm_threads
                set meta = coalesce(meta, '{}'::jsonb) || %s::jsonb
                where id = %s;
                """,
                (
                    json.dumps(
                        {
                            "is_core_memory": True,
                            "core_memory_promoted_at": datetime.utcnow().isoformat() + "Z",
                        }
                    ),
                    thread_id,
                ),
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Thread not found.")
        conn.commit()
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


def _build_memory_context(
    thread_id: str,
    latest_user_text: str,
    short_term_limit: int | None,
    reaction_user_id: str | None = None,
) -> list[dict[str, str]]:
    context: list[dict[str, str]] = []

    if SUMMARY_CUE_REGEX.search(latest_user_text):
        summary = _get_active_summary(thread_id)
        if summary:
            context.append({"role": "system", "content": f"Active summary:\n{summary}"})

    if SEMANTIC_CUE_REGEX.search(latest_user_text):
        semantic = _get_semantic_memories(thread_id, 5)
        if semantic:
            context.append(
                {
                    "role": "system",
                    "content": "Relevant long-term memory:\n" + "\n- ".join(semantic),
                }
            )

    reaction_feedback = (
        _get_recent_reaction_feedback(thread_id, reaction_user_id, 8)
        if reaction_user_id
        else []
    )
    if reaction_feedback:
        context.append(
            {
                "role": "system",
                "content": "User reaction signals:\n- " + "\n- ".join(reaction_feedback),
            }
        )

    recent = _query_events(
        thread_id=thread_id,
        limit=_normalize_limit(short_term_limit, 30, 200),
        reaction_user_id=reaction_user_id,
    )
    for message in recent:
        context.append({"role": message["role"], "content": message["content"]})

    return context


def _get_recent_reaction_feedback(
    thread_id: str, reaction_user_id: str, limit: int
) -> list[str]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select r.reaction, e.content
                from public.ltm_event_reactions r
                join public.ltm_events e on e.id = r.event_id
                where e.thread_id = %s
                  and r.user_id = %s
                  and e.actor = 'assistant'
                order by r.updated_at desc
                limit %s;
                """,
                (thread_id, reaction_user_id, limit),
            )
            rows = cur.fetchall()
        labels = {
            "thumbs_up": "liked",
            "heart": "loved",
            "angry": "disliked",
            "sad": "found unhelpful",
            "brain": "requested a summary after",
        }
        out: list[str] = []
        for reaction, content in rows:
            reaction_key = str(reaction).strip() if reaction else ""
            excerpt = str(content or "").strip().replace("\n", " ")
            if len(excerpt) > 120:
                excerpt = excerpt[:120] + "..."
            label = labels.get(reaction_key, reaction_key or "reacted")
            if excerpt:
                out.append(f"User {label}: \"{excerpt}\"")
            else:
                out.append(f"User reaction: {label}")
        return out
    finally:
        conn.close()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/threads")
def create_thread_route(
    payload: ThreadCreateRequest,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    auth_user_id = _authorize_request(x_api_key, authorization)
    user_id = _resolve_effective_user_id(payload.user_id, auth_user_id)
    thread_id = create_thread(user_id, payload.title)
    return {"thread_id": thread_id, "user_id": user_id}


@app.get("/v1/threads")
def list_threads_route(
    user_id: str | None = Query(default=None),
    limit: int = Query(50),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    auth_user_id = _authorize_request(x_api_key, authorization)
    effective_user_id = _resolve_effective_user_id(user_id, auth_user_id)
    rows = _query_threads(user_id=effective_user_id, limit=_normalize_limit(limit, 50, 200))
    return {"threads": rows, "user_id": effective_user_id}


@app.get("/v1/threads/{thread_id}/events")
def list_events_route(
    thread_id: str,
    limit: int = Query(100),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    auth_user_id = _authorize_request(x_api_key, authorization)
    _assert_thread_owner(thread_id, auth_user_id)
    reaction_user_id = _resolve_reaction_user_id(thread_id, auth_user_id)
    events = _query_events(
        thread_id=thread_id,
        limit=_normalize_limit(limit, 100, 200),
        reaction_user_id=reaction_user_id,
    )
    return {"thread_id": thread_id, "messages": events}


@app.patch("/v1/threads/{thread_id}")
def rename_thread_route(
    thread_id: str,
    payload: ThreadRenameRequest,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    auth_user_id = _authorize_request(x_api_key, authorization)
    _assert_thread_owner(thread_id, auth_user_id)
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required.")
    _rename_thread(thread_id, title)
    return {"thread_id": thread_id, "title": title, "ok": True}


@app.delete("/v1/threads/{thread_id}")
def delete_thread_route(
    thread_id: str,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    auth_user_id = _authorize_request(x_api_key, authorization)
    deleted = _delete_thread(thread_id, auth_user_id)
    return {"thread_id": thread_id, "ok": True, "deleted": deleted}


@app.post("/v1/threads/{thread_id}/promote-core-memory")
def promote_thread_core_memory_route(
    thread_id: str,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    auth_user_id = _authorize_request(x_api_key, authorization)
    _assert_thread_owner(thread_id, auth_user_id)

    summary_updated = force_update_summary(thread_id)
    _mark_thread_core_memory(thread_id)
    summary = _get_active_summary(thread_id)

    return {
        "thread_id": thread_id,
        "summary": summary,
        "summary_updated": summary_updated,
        "is_core_memory": True,
        "ok": True,
    }


@app.post("/v1/threads/{thread_id}/events")
def create_event_route(
    thread_id: str,
    payload: EventCreateRequest,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    auth_user_id = _authorize_request(x_api_key, authorization)
    _assert_thread_owner(thread_id, auth_user_id)
    event_id = add_event(
        thread_id=thread_id,
        actor=payload.actor,
        content=payload.content,
        meta=payload.meta or {},
    )
    return {"event_id": event_id}


@app.post("/v1/threads/{thread_id}/events/{event_id}/reaction")
def set_event_reaction_route(
    thread_id: str,
    event_id: str,
    payload: EventReactionRequest,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    auth_user_id = _authorize_request(x_api_key, authorization)
    _assert_thread_owner(thread_id, auth_user_id)
    normalized_event_id = _normalize_reaction_event_id(event_id)
    reaction_user_id = _resolve_reaction_user_id(thread_id, auth_user_id)
    if not reaction_user_id:
        raise HTTPException(status_code=400, detail="Unable to resolve reaction user.")

    reaction = payload.reaction.strip() if isinstance(payload.reaction, str) else None
    stored_reaction, summary_updated = _set_event_reaction(
        thread_id=thread_id,
        event_id=normalized_event_id,
        user_id=reaction_user_id,
        reaction=reaction,
    )
    return {
        "thread_id": thread_id,
        "event_id": normalized_event_id,
        "reaction": stored_reaction,
        "summary_updated": summary_updated,
        "ok": True,
    }


@app.get("/v1/threads/{thread_id}/summary")
def get_summary_route(
    thread_id: str,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    auth_user_id = _authorize_request(x_api_key, authorization)
    _assert_thread_owner(thread_id, auth_user_id)
    return {"thread_id": thread_id, "summary": _get_active_summary(thread_id)}


@app.post("/v1/threads/{thread_id}/memory-context")
def build_memory_context_route(
    thread_id: str,
    payload: MemoryContextRequest,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    auth_user_id = _authorize_request(x_api_key, authorization)
    _assert_thread_owner(thread_id, auth_user_id)
    context = _build_memory_context(
        thread_id=thread_id,
        latest_user_text=payload.latest_user_text,
        short_term_limit=payload.short_term_limit,
        reaction_user_id=_resolve_reaction_user_id(thread_id, auth_user_id),
    )
    return {"thread_id": thread_id, "messages": context}


@app.post("/v1/threads/{thread_id}/chat")
def chat_route(
    thread_id: str,
    payload: ChatRequest,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Response:
    auth_user_id = _authorize_request(x_api_key, authorization)
    _assert_thread_owner(thread_id, auth_user_id)

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message text is required.")

    try:
        add_event(thread_id=thread_id, actor="user", content=text, meta={"source": "chatui"})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to persist user event: {exc}")

    try:
        context = _build_memory_context(
            thread_id=thread_id,
            latest_user_text=text,
            short_term_limit=payload.short_term_limit,
            reaction_user_id=_resolve_reaction_user_id(thread_id, auth_user_id),
        )
        if (
            context
            and context[-1].get("role") == "user"
            and context[-1].get("content", "").strip() == text
        ):
            context = context[:-1]
        assistant_text = chat_reply(user_text=text, context_messages=context)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate assistant reply: {exc}")

    if assistant_text.strip():
        try:
            add_event(
                thread_id=thread_id,
                actor="assistant",
                content=assistant_text,
                meta={"source": "chatui_llm"},
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to persist assistant event: {exc}"
            )

    return Response(content=assistant_text, media_type="text/plain; charset=utf-8")
