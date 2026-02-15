-- CortexLTM - Event Reactions
-- Run this after 03_summaries.sql

create table if not exists public.ltm_event_reactions (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  -- Scope reaction to one assistant event.
  event_id uuid not null references public.ltm_events(id) on delete cascade,

  -- Reactions are user-specific to preserve personal preference signals.
  user_id uuid not null,

  -- Supported reaction set used by CortexUI.
  reaction text not null check (
    reaction in ('thumbs_up', 'heart', 'angry', 'sad', 'brain')
  ),

  meta jsonb not null default '{}'::jsonb
);

-- One reaction per user per event (updated in-place on change).
create unique index if not exists ltm_event_reactions_event_user_uidx
  on public.ltm_event_reactions (event_id, user_id);

create index if not exists ltm_event_reactions_user_created_idx
  on public.ltm_event_reactions (user_id, created_at desc);

create index if not exists ltm_event_reactions_event_updated_idx
  on public.ltm_event_reactions (event_id, updated_at desc);
