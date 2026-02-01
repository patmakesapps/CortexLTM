# cortexltm/embeddings.py

# We keep this simple + swappable later in case you want to use a local embedding model:
# - One function: embed_text()
# - Reads env vars
# - Uses official OpenAI SDK
# - Returns a 1536-dim list[float]

import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_DEFAULT_MODEL = "text-embedding-3-small"

# Guardrails:
# Embedding models support up to ~8192 input tokens; we can't reliably token-count
# without adding dependencies, so we cap by characters as a cheap safety measure.
_MAX_CHARS = 20000

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    _client = OpenAI(api_key=api_key)
    return _client


def embed_text(text: str) -> List[float]:
    """
    Create a 1536-dimension embedding for the given text using OpenAI.

    - Requires OPENAI_API_KEY in env
    - Uses OPENAI_EMBED_MODEL if set; defaults to text-embedding-3-small (1536 dims)
    - Trims/guards against extremely long input
    """
    if text is None:
        raise ValueError("embed_text: text is None")

    t = str(text).strip()
    if not t:
        raise ValueError("embed_text: text is empty")

    if len(t) > _MAX_CHARS:
        t = t[:_MAX_CHARS]

    model = os.getenv("OPENAI_EMBED_MODEL", "").strip() or _DEFAULT_MODEL

    client = _get_client()
    resp = client.embeddings.create(
        model=model,
        input=t,
    )

    emb = resp.data[0].embedding

    # Hard assert to match your DB column: vector(1536)
    if not isinstance(emb, list) or len(emb) != 1536:
        raise RuntimeError(
            f"Unexpected embedding size: got {len(emb) if isinstance(emb, list) else type(emb)}; expected 1536"
        )

    # Ensure floats (OpenAI returns floats, but be defensive)
    return [float(x) for x in emb]
