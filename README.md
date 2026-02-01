## CortexLTM
Schema-driven long-term memory layer for LLMs and agents.

Make sure you have/are on:  
`winget install -e --id Python.Python.3.12`  
64 bit not 32 bit

## Basic Project Setup

1) In your repo, create a Python venv by running -

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install the Groq SDK or similar -

```powershell
pip install groq
```

3) If using .env install -

```powershell
pip install python-dotenv
```

4) Refer to `.env.example`

5) Create `groq_test.py` (or similar) to load .env and test you get a response. Run with -

```powershell
python groq_test.py
```

6) if you have issues with unresolved imports do this -

Press Ctrl + Shift + P  
Type: Python: Select Interpreter  
Pick: `C:\myproject\.venv\Scripts\python.exe` (or whatever your project path is).

7) Install the DB driver in the same activated venv terminal -

```powershell
pip install psycopg2-binary ----- version 2.9.11
```

8) Install openai SDK for embedding model -

```powershell
pip install openai
```

---

## Run Scripts in sql folder

Scripts are numbered in the order they were ran. It is highly recommended to run them in the exact order as they are listed.

---

## Memory Architecture (Current Behavior)

CortexLTM now supports:

- Thread-level conversations (`ltm_threads`)
- Event-level message logging (`ltm_events`)
- Rolling summaries + episode memory (`ltm_thread_summaries`)
- Semantic search over embedded summaries and/or events
- Automatic summary updates after meaningful conversational turns

### Events

Every message (user / assistant / system) is stored in `ltm_events`.

You may optionally embed individual events using:

```python
add_event(..., embed=True)
```

If `embed=False`, the `embedding` column remains `NULL`.

Event-level embeddings are useful for fine-grained semantic search but are not required for summaries to function.

---

## Rolling Summaries (Automatic)

CortexLTM automatically builds a rolling summary per thread.

Behavior:

- A **TURN** = user message + next assistant message.
- A turn is evaluated for "meaningfulness".
- After **10 meaningful turns**, the system:
  - Builds or updates a rolling summary.
  - Embeds the summary using OpenAI.
  - Stores it in `ltm_thread_summaries`.
- If a significant topic shift is detected (via cosine similarity of summary embeddings),
  the current summary is archived and a new episode begins.

Exactly **one active summary** exists per thread at any time.

---

## Embeddings (Required for Semantic Memory)

CortexLTM uses OpenAI embeddings (`text-embedding-3-small`) for:

- Summary embeddings (default behavior)
- Optional event-level embeddings
- Topic-shift detection via cosine similarity
- Semantic search

Embeddings are generated automatically when:

- A rolling summary is created or updated
- `add_event(..., embed=True)` is used

If embeddings are disabled or fail:

- Summary rows can still be written (if configured to allow it)
- Semantic search will not return results for rows where `embedding IS NULL`

---

## Semantic Search Example

```python
from cortexltm.messages import create_thread, add_event, search_events_semantic

thread_id = create_thread("Demo Thread")

add_event(thread_id, "user", "I like sci-fi shooters and robots.", embed=True)
add_event(thread_id, "assistant", "Noted.", embed=False)

results = search_events_semantic(
    "what does the user like?",
    k=5,
    thread_id=thread_id
)

print(results)
```

---

## Current System Capabilities

- Meaningful turn detection
- Automatic rolling summary updates
- Topic shift episode detection
- Summary embedding + similarity comparison
- Event-level optional embedding
- Character limit guard in CLI
- Fully modular embedding layer (swappable later for local models)

---

CortexLTM is now operating as a structured, schema-driven long-term memory system — not just a message logger.
