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


def assistant_llm(thread_id: str, user_text: str) -> str:
    # 1) short-term context (last N messages)
    context = _fetch_recent_context(thread_id, limit=20)

    # context already includes the user's message because we log before calling this,
    # so drop the last user message from context to avoid duplication.
    if context and context[-1]["role"] == "user":
        context = context[:-1]

    # 2) memory context (thread summary + master memory)
    memory_msgs = []

    # (a) active thread summary (episode memory)
    try:
        from cortexltm.summaries import _get_active_summary  # simple v1 internal import

        active = _get_active_summary(thread_id)
        if active and (active.get("summary") or "").strip():
            memory_msgs.append(
                {
                    "role": "system",
                    "content": "THREAD SUMMARY (most recent durable context):\n"
                    + active["summary"].strip(),
                }
            )
    except Exception:
        pass

    # (b) master memory (cross-chat user memory)
    try:
        from cortexltm.master_memory import list_master_items

        # fetch user_id for this thread
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

        if user_id:
            items = list_master_items(user_id=user_id, limit=5, status="active")
            if items:
                lines = []
                for it in items:
                    bucket = it.get("bucket")
                    text = (it.get("text") or "").strip()
                    conf = it.get("confidence")
                    reinf = it.get("reinforcement_count")
                    if text:
                        lines.append(
                            f"- [{bucket}] {text} (conf={conf:.2f}, r={reinf})"
                        )
                if lines:
                    memory_msgs.append(
                        {
                            "role": "system",
                            "content": "MASTER MEMORY (cross-chat durable facts):\n"
                            + "\n".join(lines),
                        }
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
