-- CortexLTM - Connected Accounts (Provider-Agnostic OAuth Tokens)
-- Run this AFTER 04_master_memory.sql on a fresh database.
--
-- Purpose:
-- - Store per-user connected app credentials for agent tools
-- - Support Google now (Calendar/Gmail/Drive), plus non-Google providers later

-- Keep helper available even if this file is run independently.
create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create table if not exists public.ltm_connected_accounts (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  -- Internal user identity from host app auth
  user_id uuid not null,

  -- Provider slug: google, microsoft, notion, slack, etc
  provider text not null,

  -- External account identifier from provider (subject/user id/email handle)
  provider_account_id text not null,

  -- OAuth credential fields
  access_token text,
  refresh_token text,
  token_type text,
  scope text,
  expires_at timestamptz,

  -- active | revoked | expired | error
  status text not null default 'active',

  -- Provider-specific extras (tenant id, granted scopes array, account email, etc)
  meta jsonb not null default '{}'::jsonb,

  -- Soft-delete to preserve audit trails/history
  deleted_at timestamptz,

  constraint ltm_connected_accounts_provider_chk check (
    provider = lower(provider) and length(trim(provider)) > 0
  ),
  constraint ltm_connected_accounts_provider_account_id_chk check (
    length(trim(provider_account_id)) > 0
  ),
  constraint ltm_connected_accounts_status_chk check (
    status in ('active', 'revoked', 'expired', 'error')
  )
);

drop trigger if exists trg_ltm_connected_accounts_updated_at on public.ltm_connected_accounts;
create trigger trg_ltm_connected_accounts_updated_at
before update on public.ltm_connected_accounts
for each row execute function public.set_updated_at();

-- One live row per (user, provider, provider account)
create unique index if not exists ltm_connected_accounts_user_provider_account_unique_live
  on public.ltm_connected_accounts (user_id, provider, provider_account_id)
  where deleted_at is null;

-- Fast lookup: all provider connections for a user
create index if not exists ltm_connected_accounts_user_provider_idx
  on public.ltm_connected_accounts (user_id, provider);

-- Fast lookup: refresh/expiry checks
create index if not exists ltm_connected_accounts_expires_at_idx
  on public.ltm_connected_accounts (expires_at)
  where deleted_at is null;

-- Fast lookup: active/live accounts by status
create index if not exists ltm_connected_accounts_status_idx
  on public.ltm_connected_accounts (status)
  where deleted_at is null;
