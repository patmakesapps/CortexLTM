-- CortexLTM - Thread Summaries (Rolling + Episodes)
-- Run this after 02_events.sql

create table if not exists public.ltm_thread_summaries (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),

  -- Link a summary to a thread
  thread_id uuid not null references public.ltm_threads(id) on delete cascade,

  -- The span of events this summary covers (optional but very useful)
  range_start_event_id uuid references public.ltm_events(id) on delete set null,
  range_end_event_id uuid references public.ltm_events(id) on delete set null,

  -- The summary text itself
  summary text not null,

  -- Extra metadata: topic, model, tokens, reason ("topic_shift"), etc.
  meta jsonb not null default '{}'::jsonb,

  -- Optional: embedding for semantic search over summaries/episodes
  embedding vector(1536),

  -- Exactly one "active" rolling summary per thread
  is_active boolean not null default false
);

-- Fast: fetch active summary for a thread
create index if not exists ltm_thread_summaries_thread_active_idx
  on public.ltm_thread_summaries (thread_id, is_active);

-- Fast: list summaries newest-first per thread
create index if not exists ltm_thread_summaries_thread_created_at_idx
  on public.ltm_thread_summaries (thread_id, created_at desc);

-- Enforce: at most ONE active summary per thread
create unique index if not exists ltm_thread_summaries_one_active_per_thread
  on public.ltm_thread_summaries (thread_id)
  where is_active = true;
