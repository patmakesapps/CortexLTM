from cortexltm.messages import create_thread, add_event
from cortexltm.db import get_conn
from cortexltm.llm import chat_reply

MAX_USER_CHARS = 2000


def _fetch_recent_context(thread_id: str, limit: int = 16):
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
    context = _fetch_recent_context(thread_id, limit=16)
    # context already includes the user's message because we log before calling this,
    # so drop the last user message from context to avoid duplication.
    if context and context[-1]["role"] == "user":
        context = context[:-1]

    return chat_reply(user_text=user_text, context_messages=context)


def run_chat():
    thread_id = create_thread(title="CLI Chat Thread")

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
            thread_id = create_thread(title="CLI Chat Thread")
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
            meta={"source": "cli_stub"},
            importance_score=0,
        )

        print(f"bot> {reply}")

    print("\nDone.")


if __name__ == "__main__":
    run_chat()
