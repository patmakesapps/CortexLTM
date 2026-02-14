# Contributing

CortexLTM is early-stage and evolving quickly. Small, focused PRs are preferred.

## High-impact Areas

- Provider abstraction (local embeddings)
- Retrieval orchestration
- Tests
- Packaging as pip module
- Performance improvements

## Local Dev Expectations

- Use Python `3.12` and a project-local virtual environment.
- Copy `.env.example` to `.env` and set required keys before API/CLI testing.
- Run SQL migrations in numeric order from `sql/`.
- Keep backward compatibility for `/v1/*` routes unless the change is explicitly breaking.

## PR Checklist

- Include a short problem statement and scope in the PR description.
- Update docs (`README.md` and/or this file) when behavior/config changes.
- Add or adjust tests when practical; if skipped, state why.
- Keep schema changes additive unless a migration path is documented.

Open an issue before large PRs.
