-- CortexLTM - Threads
-- Run this second

-- Create the threads table. This is a container table for conversation/session.
create table if not exists public.ltm_threads (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),

  -- REQUIRED: identity key for cross-chat master memory
  user_id uuid not null,

  -- Optional: easy to read title you can set later.
  title text,

  -- Optional: you can store extra info in here...(app version, user_id, notes, etc)
  meta jsonb not null default '{}'::jsonb,

  -- Optional: use for an embedding representing the entire thread or a summary embedding.
  -- 1536 is common for many embedding models; change if your model differs.
  embedding vector(1536)
);

-- Helpful index for sorting by newest
create index if not exists ltm_threads_created_at_idx
  on public.ltm_threads (created_at desc);

-- Helpful index for fetching threads for a user
create index if not exists ltm_threads_user_created_at_idx
  on public.ltm_threads (user_id, created_at desc);
