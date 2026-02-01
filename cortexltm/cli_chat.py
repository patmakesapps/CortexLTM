from cortexltm.messages import create_thread, add_event


def assistant_stub(user_text: str) -> str:
    t = user_text.strip()
    if not t:
        return "(stub) say something and I’ll log it."
    low = t.lower()
    if low in {"hi", "hello"}:
        return "(stub) hey — logging this thread in CortexLTM."
    if low.endswith("?"):
        return "(stub) good question. (later an LLM will answer this)"
    return f"(stub) got it: {t}"


def run_chat():
    thread_id = create_thread(title="CLI Chat Thread")

    print("\n=== CortexLTM CLI Chat (STUB) ===")
    print(f"Thread ID: {thread_id}")
    print("Type messages and press Enter.")
    print("Commands: /new (new thread), /thread (show id), /exit\n")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n/exiting")
            break

        if not user_text:
            continue

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

        # generate + write assistant stub event
        reply = assistant_stub(user_text)
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
