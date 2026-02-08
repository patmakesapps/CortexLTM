-- CortexLTM - Events
-- Run this third

-- Create events table...(these are the actual messages, user/ai etc)
create table if not exists public.ltm_events(
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),

  -- Link an event to a thread (conversation/session)
  thread_id uuid not null references public.ltm_threads(id) on delete cascade,

  -- who produced the event: user, ai extract
  actor text not null,

  -- The actual message text
  content text not null,

  -- Extra metadata (optional) 
  meta jsonb not null default '{}'::jsonb,

  -- Scoring knob (0 = normal, higher = more important)
  importance_score smallint not null default 0,

  -- Optional: embedding for this specific message (or chunk). Change number inside () if your embedding model differs.
  embedding vector(1536) 
);

-- Indexes to keep the “last N messages in a thread” fast
create index if not exists ltm_events_thread_created_at_idx
  on public.ltm_events (thread_id, created_at desc);

-- Index for filtering by actor inside a thread (optional but cheap/useful)
create index if not exists ltm_events_thread_actor_created_at_idx
  on public.ltm_events (thread_id, actor, created_at desc);

-- Index for “show me important items first” in a thread (optional)
create index if not exists ltm_events_thread_importance_score_created_at_idx
  on public.ltm_events (thread_id, importance_score desc, created_at desc);

-- pgvector ANN index for semantic search (ORDER BY embedding <-> query)
create index if not exists ltm_events_embedding_ivfflat_idx
  on public.ltm_events
  using ivfflat (embedding vector_l2_ops)
  with (lists = 100)
  where embedding is not null;
