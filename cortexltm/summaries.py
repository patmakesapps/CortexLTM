# cortexltm/summaries.py
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .llm import summarize_update

from .db import get_conn
from .embeddings import embed_text

logger = logging.getLogger(__name__)

# --- v1 knobs (keep simple) ---
MEANINGFUL_TARGET = 12  # lower threshold so shorter sessions produce summaries
FETCH_LOOKBACK = 120  # pull up to N events since last summary end
TOPIC_SHIFT_COSINE_MIN = 0.75  # lower => more likely new topic
MIN_SUMMARY_UPDATE_SECONDS = 180  # debounce summary writes/embeddings


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_ts(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    try:
        s = str(raw).strip()
        if not s:
            return None
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _should_debounce_summary(active_summary: Optional[Dict[str, Any]]) -> bool:
    if not active_summary:
        return False

    meta = active_summary.get("meta") if isinstance(active_summary, dict) else None
    if not isinstance(meta, dict):
        return False

    last_write_at = _parse_iso_ts(meta.get("last_summary_write_at"))
    if not last_write_at:
        return False

    age = datetime.now(timezone.utc) - last_write_at
    return age < timedelta(seconds=MIN_SUMMARY_UPDATE_SECONDS)


def _sync_master_from_active_summary(thread_id: str) -> None:
    """
    v1 wiring: when a thread summary is updated, write ONE master-memory item
    + evidence pointing at the summary_id.

    Keep it intentionally dumb for now (no extraction):
      - bucket: PROJECTS
      - text: compact slice of the active summary
    """
    try:
        from .master_memory import upsert_master_item, add_master_evidence

        active = _get_active_summary(thread_id)
        if not active:
            return

        summary_id = active.get("id")
        summary_text = (active.get("summary") or "").strip()
        if not summary_id or not summary_text:
            return

        # Get user_id for this thread (needed for master items)
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

        if not user_id:
            return

        # Stable text key keeps this to one cross-chat item per thread.
        # Put the changing summary excerpt in meta to avoid duplicate upserts.
        compact = summary_text.replace("\n", " ").strip()
        if len(compact) > 220:
            compact = compact[:220] + "…"

        master_text = f"Thread summary reference ({thread_id})"

        master_id = upsert_master_item(
            user_id=user_id,
            bucket="LONG_RUNNING_CONTEXT",
            text=master_text,
            confidence=0.55,
            stability="med",
            embed=False,
            meta={
                "source": "auto_summary_wire",
                "summary_id": str(summary_id),
                "summary_excerpt": compact,
            },
        )

        add_master_evidence(
            master_item_id=master_id,
            thread_id=thread_id,
            summary_id=summary_id,
            weight=1.0,
            meta={"source": "auto_summary_wire"},
        )
    except Exception:
        # Never let master-memory wiring break the chat loop
        logger.exception("summary->master sync failed for thread_id=%s", thread_id)
        return


# -----------------------------
# helpers
# -----------------------------
def _vector_literal(vec: list[float] | None) -> str | None:
    if vec is None:
        return None
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    # no numpy; keep it tiny
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return (dot / denom) if denom else 0.0


def is_meaningful_turn(user_text: str, assistant_text: str) -> bool:
    """
    v1 "meaningful" heuristic applied to a TURN (user + next assistant).

    The goal:
    - skip greetings/filler/back-and-forth with no durable info
    - count anything that looks like a decision, plan, question, constraint, preference, or detail

    Keep intentionally simple; we'll refine later.
    """
    u = "" if user_text is None else str(user_text).strip()
    a = "" if assistant_text is None else str(assistant_text).strip()

    if not u and not a:
        return False

    combo = (u + "\n" + a).strip()
    low_u = u.lower().strip()

    trivial_user = low_u in {
        "ok",
        "okay",
        "k",
        "kk",
        "lol",
        "lmao",
        "bet",
        "nice",
        "cool",
        "word",
        "yup",
        "yeah",
        "nah",
        "thanks",
        "thx",
        "ty",
        "cya",
        "bye",
        "goodbye",
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "what's up",
        "whats up",
    }

    # if user is trivial AND assistant is also short => not meaningful
    if trivial_user and len(a) < 80:
        return False

    # if the combined turn is extremely short => not meaningful
    if len(combo) < 40:
        return False

    # strong signals (user side)
    u_low = u.lower()

    strong_phrases = [
        # Identity / profile
        "my name is",
        "call me ",
        "i am ",
        "i'm ",
        "i live",
        "i work",
        "my email",
        "my phone",
        "my address",
        "my birthday",
        # Third-person identity (sometimes important in convo)
        "her name",
        "his name",
        "their name",
        # Relationships / entities
        "my dog",
        "my cat",
        "my pet",
        "my boss",
        "my brother",
        "my sister",
        "my mother",
        "my mom",
        "my dad",
        "my father",
        "my husband",
        "my wife",
        "my kid",
        "my boyfriend",
        "my girlfriend",
        "my friend",
        # Preferences / habits
        "i like",
        "i love",
        "i hate",
        "i prefer",
        "my favorite",
        "i don't like",
        "i dislike",
        "i usually",
        "i always",
        "i never",
        # Memory intent
        "remember",
        # Commitments / plans
        "i will",
        "i want to",
        "i need to",
        "we need to",
        "we should",
        "let's",
        "let's do",
        "we will",
        "next step",
        "goal is",
        "deadline",
        "plan",
        # Constraints / rules
        "do not",
        "don't",
        "never ",
        "avoid ",
        "stop ",
        "no more",
        "only ",
        # Project / decisions (dev-oriented, high-signal)
        "use ",
        "build ",
        "implement ",
        "schema",
        "table",
        "sql",
        "api",
        "bug",
        "error",
    ]

    if any(p in u_low for p in strong_phrases):
        return True

    # otherwise: if it has some substance, count it
    has_alnum = any(ch.isalnum() for ch in combo)
    return has_alnum and len(combo) >= 80


def _get_active_summary(thread_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select id, summary, meta, range_start_event_id, range_end_event_id
                from public.ltm_thread_summaries
                where thread_id = %s and is_active = true
                limit 1;
                """,
                (thread_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            id_, summary, meta, start_id, end_id = row
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {"_raw": meta}
            return {
                "id": str(id_),
                "summary": summary,
                "meta": meta,
                "range_start_event_id": str(start_id) if start_id else None,
                "range_end_event_id": str(end_id) if end_id else None,
            }
    finally:
        conn.close()


def _get_event_created_at(event_id: str) -> Optional[str]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select created_at from public.ltm_events where id = %s limit 1;",
                (event_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def _fetch_events_since(
    thread_id: str, since_event_id: Optional[str]
) -> List[Dict[str, Any]]:
    since_created_at = None
    if since_event_id:
        since_created_at = _get_event_created_at(since_event_id)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if since_created_at:
                cur.execute(
                    """
                    select id, created_at, actor, content
                    from public.ltm_events
                    where thread_id = %s
                      and created_at > %s
                    order by created_at asc
                    limit %s;
                    """,
                    (thread_id, since_created_at, FETCH_LOOKBACK),
                )
            else:
                cur.execute(
                    """
                    select id, created_at, actor, content
                    from public.ltm_events
                    where thread_id = %s
                    order by created_at asc
                    limit %s;
                    """,
                    (thread_id, FETCH_LOOKBACK),
                )
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for id_, created_at, actor, content in rows:
            out.append(
                {
                    "id": str(id_),
                    "created_at": created_at,
                    "actor": actor,
                    "content": content,
                }
            )
        return out
    finally:
        conn.close()


def _build_candidate_summary(
    prior_summary: Optional[str], turn_lines: List[str]
) -> str:
    """
    v2: use Groq to keep the rolling summary concise.
    """
    try:
        return summarize_update(prior_summary, turn_lines)
    except Exception as e:
        # fallback: keep old behavior if LLM is unavailable
        update_lines = []
        for line in turn_lines:
            s = str(line).strip().replace("\n", " ")
            if len(s) > 260:
                s = s[:260] + "…"
            update_lines.append(f"- {s}")

        if not prior_summary:
            return "Summary so far:\n" + "\n".join(update_lines)

        return prior_summary.rstrip() + "\n\nNew info:\n" + "\n".join(update_lines)


def _archive_active_summary(thread_id: str) -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.ltm_thread_summaries
                set is_active = false
                where thread_id = %s and is_active = true;
                """,
                (thread_id,),
            )
        conn.commit()
    finally:
        conn.close()


def _insert_active_summary(
    thread_id: str,
    summary_text: str,
    start_event_id: Optional[str],
    end_event_id: Optional[str],
    meta: Dict[str, Any],
    embed: bool = True,
    summary_embedding: Optional[List[float]] = None,
) -> str:
    emb = None
    if embed:
        emb = (
            summary_embedding
            if summary_embedding is not None
            else embed_text(summary_text)
        )
    emb_literal = _vector_literal(emb)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.ltm_thread_summaries
                  (thread_id, range_start_event_id, range_end_event_id, summary, meta, embedding, is_active)
                values
                  (%s, %s, %s, %s, %s::jsonb,
                   case when %s is null then null else (%s)::vector end,
                   true)
                returning id;
                """,
                (
                    thread_id,
                    start_event_id,
                    end_event_id,
                    summary_text,
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


def _update_active_summary(
    thread_id: str,
    summary_text: str,
    end_event_id: Optional[str],
    meta_patch: Dict[str, Any],
    embed: bool = True,
    summary_embedding: Optional[List[float]] = None,
) -> None:
    emb = None
    if embed:
        emb = (
            summary_embedding
            if summary_embedding is not None
            else embed_text(summary_text)
        )
    emb_literal = _vector_literal(emb)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.ltm_thread_summaries
                set
                  summary = %s,
                  range_end_event_id = %s,
                  meta = meta || (%s::jsonb),
                  embedding = (%s)::vector
                where thread_id = %s and is_active = true;
                """,
                (
                    summary_text,
                    end_event_id,
                    json.dumps(meta_patch),
                    emb_literal,
                    thread_id,
                ),
            )
        conn.commit()
    finally:
        conn.close()


# -----------------------------
# main entrypoint (called after events are written)
# -----------------------------
def maybe_update_summary(thread_id: str) -> bool:
    """
    v1 behavior:
    - Count meaningful TURNS since last summary end.
      TURN = user event + next assistant event (if present).
    - When we have MEANINGFUL_TARGET meaningful turns:
        - Build candidate summary text (simple heuristic)
        - Topic-shift check using cosine similarity of summary embeddings
        - If shift: archive old active + insert new active
        - Else: update active rolling summary
    Returns True if we wrote/updated a summary row.
    """
    active = _get_active_summary(thread_id)
    if _should_debounce_summary(active):
        logger.debug("summary update debounced for thread_id=%s", thread_id)
        return False

    since_event_id = active["range_end_event_id"] if active else None

    events = _fetch_events_since(thread_id, since_event_id)

    meaningful_turns: List[Dict[str, Any]] = []
    i = 0
    while i < len(events):
        e = events[i]
        if e["actor"] != "user":
            i += 1
            continue

        user_event = e
        assistant_event = None

        # next assistant event after this user event (if present)
        j = i + 1
        while j < len(events):
            if events[j]["actor"] == "assistant":
                assistant_event = events[j]
                break
            j += 1

        u = user_event["content"] or ""
        a = assistant_event["content"] if assistant_event else ""

        if is_meaningful_turn(u, a):
            meaningful_turns.append(
                {
                    "user_id": user_event["id"],
                    "assistant_id": assistant_event["id"] if assistant_event else None,
                    "user_text": u,
                    "assistant_text": a,
                }
            )
            if len(meaningful_turns) >= MEANINGFUL_TARGET:
                break

        i += 1
    logger.debug(
        "meaningful_turns=%s/%s for thread_id=%s",
        len(meaningful_turns),
        MEANINGFUL_TARGET,
        thread_id,
    )

    if len(meaningful_turns) < MEANINGFUL_TARGET:
        return False

    start_event_id = meaningful_turns[0]["user_id"]
    last = meaningful_turns[-1]
    end_event_id = last["assistant_id"] or last["user_id"]

    # build summary input as compact turn lines
    turn_lines: List[str] = []
    for t in meaningful_turns:
        u_line = str(t["user_text"]).strip().replace("\n", " ")
        a_line = str(t["assistant_text"]).strip().replace("\n", " ")
        if len(u_line) > 220:
            u_line = u_line[:220] + "…"
        if len(a_line) > 220:
            a_line = a_line[:220] + "…"
        if a_line:
            turn_lines.append(f"USER: {u_line} | ASSISTANT: {a_line}")
        else:
            turn_lines.append(f"USER: {u_line}")

    prior_summary = active["summary"] if active else None
    candidate = _build_candidate_summary(prior_summary, turn_lines)

    # decide topic shift (v1 = embedding similarity)
    topic_shift = False
    candidate_emb: Optional[List[float]] = None
    if prior_summary:
        try:
            a_emb = embed_text(prior_summary)
            candidate_emb = embed_text(candidate)
            sim = _cosine_similarity(a_emb, candidate_emb)
            topic_shift = sim < TOPIC_SHIFT_COSINE_MIN
        except Exception as e:
            # if embeddings fail, fall back to "no shift" to avoid fragmentation
            logger.warning(
                "topic-shift embedding check failed for thread_id=%s: %s: %s",
                thread_id,
                type(e).__name__,
                e,
            )
            topic_shift = False

    if not active:
        _insert_active_summary(
            thread_id=thread_id,
            summary_text=candidate,
            start_event_id=start_event_id,
            end_event_id=end_event_id,
            meta={
                "reason": "init",
                "meaningful_turns": MEANINGFUL_TARGET,
                "last_summary_write_at": _utcnow_iso(),
            },
            embed=True,
            summary_embedding=candidate_emb,
        )

        # v1 wiring: write one master item from the active summary
        _sync_master_from_active_summary(thread_id)

        return True

    if topic_shift:
        _archive_active_summary(thread_id)

        # new episode starts fresh (only the new turns)
        episode_text = _build_candidate_summary(None, turn_lines)

        _insert_active_summary(
            thread_id=thread_id,
            summary_text=episode_text,
            start_event_id=start_event_id,
            end_event_id=end_event_id,
            meta={
                "reason": "topic_shift",
                "meaningful_turns": MEANINGFUL_TARGET,
                "last_summary_write_at": _utcnow_iso(),
            },
            embed=True,
        )

        _sync_master_from_active_summary(thread_id)

        return True

    _update_active_summary(
        thread_id=thread_id,
        summary_text=candidate,
        end_event_id=end_event_id,
        meta_patch={
            "reason": "rolling_update",
            "meaningful_turns": MEANINGFUL_TARGET,
            "last_summary_write_at": _utcnow_iso(),
        },
        embed=True,
        summary_embedding=candidate_emb,
    )

    return True


def force_update_summary(thread_id: str) -> bool:
    """
    Force a summary update immediately.
    - Ignores debounce windows.
    - Uses meaningful turns when available.
    - Falls back to non-trivial turns so short threads can still be summarized.
    Returns True when a summary row is inserted/updated.
    """
    active = _get_active_summary(thread_id)
    since_event_id = active["range_end_event_id"] if active else None
    events = _fetch_events_since(thread_id, since_event_id)

    turns: List[Dict[str, Any]] = []
    i = 0
    while i < len(events):
        e = events[i]
        if e["actor"] != "user":
            i += 1
            continue

        user_event = e
        assistant_event = None
        j = i + 1
        while j < len(events):
            if events[j]["actor"] == "assistant":
                assistant_event = events[j]
                break
            j += 1

        user_text = user_event["content"] or ""
        assistant_text = assistant_event["content"] if assistant_event else ""
        turns.append(
            {
                "user_id": user_event["id"],
                "assistant_id": assistant_event["id"] if assistant_event else None,
                "user_text": user_text,
                "assistant_text": assistant_text,
                "meaningful": is_meaningful_turn(user_text, assistant_text),
            }
        )
        i += 1

    if not turns:
        # Nothing new to summarize; still sync master from existing summary if available.
        _sync_master_from_active_summary(thread_id)
        return False

    meaningful_turns = [turn for turn in turns if turn["meaningful"]]
    selected_turns = meaningful_turns if meaningful_turns else turns
    selected_turns = selected_turns[: max(1, min(len(selected_turns), MEANINGFUL_TARGET))]

    start_event_id = selected_turns[0]["user_id"]
    last = selected_turns[-1]
    end_event_id = last["assistant_id"] or last["user_id"]

    turn_lines: List[str] = []
    for turn in selected_turns:
        user_line = str(turn["user_text"]).strip().replace("\n", " ")
        assistant_line = str(turn["assistant_text"]).strip().replace("\n", " ")
        if len(user_line) > 220:
            user_line = user_line[:220] + "..."
        if len(assistant_line) > 220:
            assistant_line = assistant_line[:220] + "..."
        if assistant_line:
            turn_lines.append(f"USER: {user_line} | ASSISTANT: {assistant_line}")
        else:
            turn_lines.append(f"USER: {user_line}")

    prior_summary = active["summary"] if active else None
    candidate = _build_candidate_summary(prior_summary, turn_lines)

    if not active:
        _insert_active_summary(
            thread_id=thread_id,
            summary_text=candidate,
            start_event_id=start_event_id,
            end_event_id=end_event_id,
            meta={
                "reason": "forced_promote_init",
                "forced": True,
                "meaningful_turns": len(meaningful_turns),
                "last_summary_write_at": _utcnow_iso(),
            },
            embed=True,
        )
    else:
        _update_active_summary(
            thread_id=thread_id,
            summary_text=candidate,
            end_event_id=end_event_id,
            meta_patch={
                "reason": "forced_promote_update",
                "forced": True,
                "meaningful_turns": len(meaningful_turns),
                "last_summary_write_at": _utcnow_iso(),
            },
            embed=True,
        )

    _sync_master_from_active_summary(thread_id)
    return True
