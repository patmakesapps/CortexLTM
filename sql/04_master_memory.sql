-- CortexLTM - Master Memory (Cross-Chat, User-Level)
-- Run this AFTER 03_summaries.sql on a fresh database.
--
-- Assumptions:
-- - ltm_threads already has user_id uuid NOT NULL (from 01_threads.sql)
-- - vector extension already enabled (from 00_extensions.sql)

-- -----------------------------
-- 1) updated_at trigger helper
-- -----------------------------
create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

-- -----------------------------
-- 2) Master Memory Items (atomic claims)
-- -----------------------------
create table if not exists public.ltm_master_items (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  -- Identity key (from host app auth)
  user_id uuid not null,

  -- One of the 9 v1 buckets:
  -- PROFILE, PREFERENCES, CONSTRAINTS, COMMUNICATION_STYLE, LONG_RUNNING_CONTEXT,
  -- GOALS, PROJECTS, NEXT_ACTIONS, OPEN_LOOPS
  bucket text not null,

  -- The actual memory claim (atomic)
  text text not null,

  -- active | deprecated | conflicted
  status text not null default 'active',

  -- high | med | low
  stability text not null default 'med',

  -- 0..1 confidence
  confidence double precision not null default 0.50,

  -- how often it has been reinforced across chats
  reinforcement_count integer not null default 0,

  last_seen_at timestamptz,
  last_reinforced_at timestamptz,

  -- optional structured info (conflict refs, reasons, tags, etc)
  meta jsonb not null default '{}'::jsonb,

  -- optional embedding for semantic retrieval
  embedding vector(1536),

  constraint ltm_master_items_bucket_chk check (
    bucket in (
      'PROFILE',
      'PREFERENCES',
      'CONSTRAINTS',
      'COMMUNICATION_STYLE',
      'LONG_RUNNING_CONTEXT',
      'GOALS',
      'PROJECTS',
      'NEXT_ACTIONS',
      'OPEN_LOOPS'
    )
  ),

  constraint ltm_master_items_status_chk check (
    status in ('active', 'deprecated', 'conflicted')
  ),

  constraint ltm_master_items_stability_chk check (
    stability in ('high', 'med', 'low')
  ),

  constraint ltm_master_items_confidence_chk check (
    confidence >= 0.0 and confidence <= 1.0
  )
);

drop trigger if exists trg_ltm_master_items_updated_at on public.ltm_master_items;
create trigger trg_ltm_master_items_updated_at
before update on public.ltm_master_items
for each row execute function public.set_updated_at();

-- Indexes for retrieval
create index if not exists ltm_master_items_user_bucket_status_idx
  on public.ltm_master_items (user_id, bucket, status);

create index if not exists ltm_master_items_user_updated_at_idx
  on public.ltm_master_items (user_id, updated_at desc);

create index if not exists ltm_master_items_user_confidence_idx
  on public.ltm_master_items (user_id, confidence desc);

-- pgvector ANN index for master memory semantic retrieval
create index if not exists ltm_master_items_embedding_ivfflat_idx
  on public.ltm_master_items
  using ivfflat (embedding vector_l2_ops)
  with (lists = 100)
  where embedding is not null;

-- -----------------------------
-- 3) Evidence table (audit trail)
-- -----------------------------
create table if not exists public.ltm_master_evidence (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),

  master_item_id uuid not null references public.ltm_master_items(id) on delete cascade,

  -- Evidence pointers (one or more can be set)
  thread_id uuid references public.ltm_threads(id) on delete set null,
  event_id uuid references public.ltm_events(id) on delete set null,
  summary_id uuid references public.ltm_thread_summaries(id) on delete set null,

  -- How strong this evidence is (v1: 0..1-ish, keep it loose)
  weight double precision not null default 1.0,

  -- optional notes (why linked, extraction method, etc)
  meta jsonb not null default '{}'::jsonb,

  -- At least one pointer must exist
  constraint ltm_master_evidence_pointer_chk check (
    thread_id is not null or event_id is not null or summary_id is not null
  )
);

create index if not exists ltm_master_evidence_master_item_idx
  on public.ltm_master_evidence (master_item_id);

create index if not exists ltm_master_evidence_thread_idx
  on public.ltm_master_evidence (thread_id);

create index if not exists ltm_master_evidence_event_idx
  on public.ltm_master_evidence (event_id);

create index if not exists ltm_master_evidence_summary_idx
  on public.ltm_master_evidence (summary_id);
