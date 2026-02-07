import os
from dotenv import load_dotenv

from cortexltm.messages import create_thread, add_event
from cortexltm.db import get_conn
from cortexltm.llm import chat_reply

load_dotenv(override=True)


MAX_USER_CHARS = 2000


def _fetch_recent_context(thread_id: str, limit: int = 9):
    """
    Returns messages formatted for Groq:
      [{"role":"user"|"assistant","content":"..."}]
    """
    if limit < 1:
        limit = 1
    if limit > 40:
        limit = 40

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select actor, content
                from public.ltm_events
                where thread_id = %s
                order by created_at desc
                limit %s;
                """,
                (thread_id, limit),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    # rows are newest-first, reverse to chronological
    rows.reverse()

    out = []
    for actor, content in rows:
        role = "assistant" if actor == "assistant" else "user"
        out.append({"role": role, "content": (content or "")})
    return out


def _needs_semantic_memory(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    if not t:
        return False

    cues = [
        "what's my",
        "whats my",
        "what is my",
        "do you remember",
        "remember",
        "remind me",
        "what did i say",
        "what did we say",
        "earlier you said",
        "last time",
        "my name",
        "who am i",
        "what is",
        "who is",
        "who was",
        "what was",
        "what was the plan",
        "recap",
        "summarize",
        "summary",
    ]
    return any(c in t for c in cues)


def _should_include_summary(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    if not t:
        return False

    cues = [
        "what was the plan",
        "recap",
        "summarize",
        "summary",
        "catch me up",
        "where were we",
        "continue",
        "ready to continue",
        "lets move on",
    ]
    return any(c in t for c in cues)


def _format_retrieved_block(title: str, lines: list[str]) -> dict:
    # Keep it short; this is meant to be “evidence”, not a dump.
    clamped = []
    for s in lines:
        x = (s or "").strip().replace("\n", " ")
        if not x:
            continue
        if len(x) > 220:
            x = x[:220] + "…"
        clamped.append(x)

    return {
        "role": "system",
        "content": f"{title}:\n" + "\n".join(f"- {x}" for x in clamped[:6]),
    }


def assistant_llm(thread_id: str, user_text: str) -> str:
    # 1) short-term context (last N messages)
    context = _fetch_recent_context(thread_id, limit=20)

    # context already includes the user's message because we log before calling this,
    # so drop the last user message from context to avoid duplication.
    if context and context[-1]["role"] == "user":
        context = context[:-1]

    # 2) memory context (ONLY when needed)
    memory_msgs = []

    needs_mem = _needs_semantic_memory(user_text)
    wants_summary = _should_include_summary(user_text)

    # fetch user_id once
    user_id = None
    try:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "select user_id from public.ltm_threads where id = %s limit 1;",
                    (thread_id,),
                )
                row = cur.fetchone()
            user_id = str(row[0]) if row and row[0] else None
        finally:
            conn.close()
    except Exception:
        user_id = None

    # (a) active thread summary (episode memory)  only when user asks for recap/context
    if wants_summary:
        try:
            from cortexltm.summaries import _get_active_summary  # v1 internal import

            active = _get_active_summary(thread_id)
            if active and (active.get("summary") or "").strip():
                memory_msgs.append(
                    {
                        "role": "system",
                        "content": "THREAD SUMMARY:\n" + active["summary"].strip(),
                    }
                )
        except Exception:
            pass

    # (b) semantic retrieval  only when needed
    if needs_mem and user_id:
        # 1) try MASTER memory semantic (durable)
        try:
            from cortexltm.master_memory import search_master_items_semantic

            hits = search_master_items_semantic(
                user_id=user_id, query=user_text, k=6, status="active"
            )
            lines = []
            for h in hits:
                txt = (h.get("text") or "").strip()
                bucket = h.get("bucket") or "UNKNOWN"
                if not txt:
                    continue
                lines.append(f"[{bucket}] {txt}")

            if lines:
                memory_msgs.append(
                    _format_retrieved_block(
                        "RETRIEVED MASTER MEMORY (use if relevant)", lines[:5]
                    )
                )
        except Exception:
            pass

        # 2) also try EVENT semantic (raw evidence)
        # Prefer user events and higher-importance lines to reduce junk.
        try:
            from cortexltm.messages import search_events_semantic

            ev = search_events_semantic(
                user_id=user_id,
                query=user_text,
                k=8,
                thread_id=None,  # cross-thread for this user
                only_actor="user",  # user-authored facts
                min_importance=3,  # only embedded + meaningful ones
            )

            lines = []
            for e in ev:
                txt = (e.get("content") or "").strip()
                if not txt:
                    continue
                lines.append(txt)

            if lines:
                memory_msgs.append(
                    _format_retrieved_block(
                        "RETRIEVED PAST USER FACTS (use if relevant)", lines[:4]
                    )
                )
        except Exception:
            pass

    # Put memory BEFORE the short-term chat context
    merged = memory_msgs + context

    return chat_reply(user_text=user_text, context_messages=merged)


def run_chat():
    user_id = (os.getenv("CORTEXLTM_USER_ID") or "").strip()
    if not user_id:
        raise RuntimeError(
            "Missing CORTEXLTM_USER_ID in .env (must be a uuid). "
            "Example: CORTEXLTM_USER_ID=00000000-0000-0000-0000-000000000000"
        )

    thread_id = create_thread(user_id=user_id, title="CLI Chat Thread")

    print("\n=== CortexLTM CLI Chat (STUB) ===")
    print(f"Thread ID: {thread_id}")
    print("Type messages and press Enter.")
    print("Commands: /new (new thread), /thread (show id), /exit\n")

    over_limit = False

    while True:
        try:
            user_text = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print("\n/exiting")
            break

        user_text = user_text.strip()

        if not user_text:
            continue

        # show the warning until they fix the input length
        if len(user_text) > MAX_USER_CHARS:
            over_limit = True
            print(
                f"(error) Character limit exceeded ({len(user_text)}/{MAX_USER_CHARS}). "
                f"Please shorten your message."
            )
            continue

        # once they're back under limit, clear the warning state
        if over_limit:
            over_limit = False

        if user_text in {"/exit", "/quit"}:
            break

        if user_text == "/thread":
            print(f"(info) thread_id = {thread_id}")
            continue

        if user_text == "/new":
            thread_id = create_thread(user_id=user_id, title="CLI Chat Thread")
            print(f"(info) new thread_id = {thread_id}")
            continue

        # write user event
        add_event(
            thread_id=thread_id,
            actor="user",
            content=user_text,
            meta={"source": "cli"},
            importance_score=0,
        )

        # generate + write assistant reply
        reply = assistant_llm(thread_id, user_text)
        add_event(
            thread_id=thread_id,
            actor="assistant",
            content=reply,
            meta={"source": "cli_llm"},
            importance_score=0,
        )

        print(f"bot> {reply}")

    print("\nDone.")


if __name__ == "__main__":
    run_chat()
