from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .db import get_conn
from .llm import _DEFAULT_SUMMARY_MODEL, _get_client, _model
from uuid import UUID

logger = logging.getLogger(__name__)

EXTRACTION_LIMIT = 8
CONFIDENCE_FLOOR = 0.80
BANNED_SUBSTRINGS = [
    "improve productivity",
    "explore new hobby",
    "tackle put-off tasks",
    "review previous conversation",
]


def _fetch_recent_events(
    thread_id: str, limit: int = EXTRACTION_LIMIT
) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select id, actor, content
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

    events: List[Dict[str, Any]] = []
    for id_, actor, content in rows:
        events.append(
            {
                "id": str(id_),
                "actor": actor,
                "content": content or "",
            }
        )
    return events


def _build_extraction_prompt(events: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, event in enumerate(reversed(events), 1):
        actor = (event.get("actor") or "").strip().lower()
        if actor != "user":
            continue

        content = (event.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{idx}. USER: {content}")
    if not lines:
        return ""

    return "Events:\n" + "\n".join(lines)


def _run_llm(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt_body = _build_extraction_prompt(events)
    if not prompt_body:
        return []

    messages = [
        {
            "role": "system",
            "content": (
                "You are a memory extractor.\n"
                "Extract ONLY durable USER facts, constraints, preferences, and explicit plans/commitments.\n"
                "Return JSON only (no prose).\n\n"
                "Hard rules:\n"
                "- ONLY extract from USER lines. Ignore ASSISTANT lines unless the USER explicitly accepts/commits.\n"
                "- Do NOT store generic self-help or boilerplate tasks (e.g., 'be productive', 'explore hobbies').\n"
                "- Do NOT store vague goals unless the user clearly states an ongoing objective.\n"
                "- Prefer specific nouns, names, projects, decisions, constraints, and next actions the user committed to.\n"
                "- If uncertain, return an empty array [].\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "Return a JSON array of objects with keys "
                "`text` (the fact to remember), `bucket` (PROJECTS, LONG_RUNNING_CONTEXT, PROFILE, GOALS, NEXT_ACTIONS), "
                "`confidence` (0.0-1.0), and optionally `event_id` (the originating event id). "
                "Be conservative and only include things the user clearly cares about."
                "\n\n" + prompt_body
            ),
        },
    ]

    client = _get_client()
    model = _model("GROQ_EXTRACTOR_MODEL", _DEFAULT_SUMMARY_MODEL)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=400,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        return []

    parsed: Optional[Any] = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = content[start : end + 1]
            try:
                parsed = json.loads(snippet)
            except json.JSONDecodeError as exc:
                logger.debug("extractor JSON decode failed (snippet): %s", exc)
                return []
        else:
            logger.debug("extractor JSON decode failed, no array found")
            return []

    if isinstance(parsed, list):
        return parsed
    return []


def _normalize_event_id(
    claim: Dict[str, Any], ordered_events: List[Dict[str, Any]]
) -> Optional[str]:
    raw = claim.get("event_id")
    if raw:
        try:
            return str(UUID(str(raw)))
        except (ValueError, TypeError):
            pass

    for key in ("event_index", "event_number", "idx", "position"):
        idx = claim.get(key)
        if isinstance(idx, int):
            if 1 <= idx <= len(ordered_events):
                return ordered_events[idx - 1]["id"]
        elif isinstance(idx, str) and idx.isdigit():
            numeric = int(idx)
            if 1 <= numeric <= len(ordered_events):
                return ordered_events[numeric - 1]["id"]

    return None


def extract_and_write_master_memory(*, thread_id: str, user_id: str) -> None:
    if not thread_id or not user_id:
        return

    events = _fetch_recent_events(thread_id)
    if not events:
        return

    ordered_events = list(reversed(events))

    claims = _run_llm(events)
    if not claims:
        return

    try:
        from .master_memory import upsert_master_item, add_master_evidence
    except Exception:
        logger.exception("master memory import failed")
        return

    for claim in claims:
        text = (claim.get("text") or "").strip()
        if not text:
            continue
        low = text.lower()
        if any(b in low for b in BANNED_SUBSTRINGS):
            continue
        bucket = claim.get("bucket") or "PROJECTS"
        confidence = float(claim.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        if confidence < CONFIDENCE_FLOOR:
            continue

        event_id = _normalize_event_id(claim, ordered_events)

        try:
            master_id = upsert_master_item(
                user_id=user_id,
                bucket=bucket,
                text=text,
                confidence=confidence,
                stability="med",
                embed=(confidence >= 0.90),
                meta={
                    "source": "llm_extractor",
                    "thread_id": thread_id,
                    "event_id": event_id,
                },
            )
            add_master_evidence(
                master_item_id=master_id,
                thread_id=thread_id,
                event_id=event_id,
                weight=1.0,
                meta={"source": "llm_extractor"},
            )
        except Exception:
            logger.exception("failed to write extracted master memory")
