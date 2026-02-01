# cortexltm/llm.py
import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)

_client: Groq | None = None

_DEFAULT_CHAT_MODEL = "llama-3.1-8b-instant"
_DEFAULT_SUMMARY_MODEL = "llama-3.1-8b-instant"

# cheap safety caps (chars, not tokens)
_MAX_USER_CHARS = 4000
_MAX_TURN_LINE_CHARS = 600
_MAX_CONTEXT_MESSAGES = 20


def _get_client() -> Groq:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in .env")

    _client = Groq(api_key=api_key)
    return _client


def _model(name_env: str, default_value: str) -> str:
    return os.getenv(name_env, "").strip() or default_value


def chat_reply(
    user_text: str,
    context_messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    context_messages format:
      [{"role":"user"|"assistant", "content":"..."}]
    """
    t = (user_text or "").strip()
    if not t:
        return "Say something and I’ll respond."

    if len(t) > _MAX_USER_CHARS:
        t = t[:_MAX_USER_CHARS]

    msgs: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Be concise, accurate, and direct."
                "Be engaging and confident but not annoying. Offer a solution or propose a next step if applicable."
            ),
        }
    ]

    if context_messages:
        # keep it bounded
        trimmed = context_messages[-_MAX_CONTEXT_MESSAGES:]
        msgs.extend(trimmed)

    msgs.append({"role": "user", "content": t})

    client = _get_client()
    model = _model("GROQ_CHAT_MODEL", _DEFAULT_CHAT_MODEL)

    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0.6,
        max_tokens=600,
    )

    out = (resp.choices[0].message.content or "").strip()
    return out or "Okay."


def summarize_update(
    prior_summary: Optional[str],
    turn_lines: List[str],
) -> str:
    """
    Produce a concise rolling summary.

    - prior_summary: existing summary text (optional)
    - turn_lines: list of compact "USER: ... | ASSISTANT: ..." strings
    Returns: 3-7 short bullets, stable and durable.
    """
    # clamp inputs
    cleaned_lines: List[str] = []
    for line in turn_lines:
        s = (line or "").replace("\n", " ").strip()
        if not s:
            continue
        if len(s) > _MAX_TURN_LINE_CHARS:
            s = s[:_MAX_TURN_LINE_CHARS] + "…"
        cleaned_lines.append(s)

    if not cleaned_lines:
        return (prior_summary or "").strip() or "No durable info yet."

    prior = (prior_summary or "").strip()

    prompt = (
        "Update the conversation memory summary.\n"
        "Rules:\n"
        "- Output ONLY the updated summary.\n"
        "- Use 3–7 short bullet points.\n"
        "- Capture durable facts, decisions, constraints, goals, next steps.\n"
        "- Do NOT include chit-chat, greetings, filler.\n"
        "- Do NOT invent details.\n"
        "- Prefer concrete names, numbers, and tasks.\n"
    )

    user_payload = "NEW TURNS:\n" + "\n".join(f"- {x}" for x in cleaned_lines)
    if prior:
        user_payload = "PRIOR SUMMARY:\n" + prior + "\n\n" + user_payload

    client = _get_client()
    model = _model("GROQ_SUMMARY_MODEL", _DEFAULT_SUMMARY_MODEL)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_payload},
        ],
        temperature=0.2,
        max_tokens=350,
    )

    out = (resp.choices[0].message.content or "").strip()
    return out or (prior if prior else "No durable info yet.")
