# AGENTS.md

## Purpose

This repository is a Python 3.12+ analysis library for the cmudrc design
research ecosystem. Keep changes focused, keep the public API intentional, and
preserve reproducible table-analysis workflows across sequence, language,
dimensionality-reduction, and statistical pipelines.

## Setup

- Create and activate a virtual environment:
  - `python -m venv .venv`
  - `source .venv/bin/activate`
- The preferred interpreter target lives in `.python-version` (`3.12`).
- Install local tooling with `make dev`.

## Testing And Validation

Use the smallest useful check while iterating, then run the full gate before
merging.

- Fast local loop:
  - `make fmt`
  - `make lint`
  - `make type`
  - `make test`
- If docs changed:
  - `make docs-check`
  - `make docs`
- If the example changed:
  - `make run-examples`
  - `make examples-coverage`
- Pre-merge baseline:
  - `make ci`
- Pre-publish baseline:
  - `make release-check` (builds artifacts, validates metadata, and smoke-installs the wheel)

## Public Vs Private Boundaries

- The supported public surface is whatever is re-exported from
  `src/design_research_analysis/__init__.py`.
- Prefer adding new public behavior to stable top-level modules before creating
  deeper internal package trees.
- If you add internal helper modules later, prefix them with `_` and keep them
  out of the top-level exports unless there is a deliberate API decision.

## Behavioral Guardrails

- Keep tests deterministic and offline by default.
- Update tests, docs, and examples alongside behavior changes.
- Avoid broad dependency growth in the base install.
- Keep CLI outputs and reproducibility metadata stable unless a change
  explicitly updates the release contract.

## Release Planning

- Do not create monthly milestone naming tables, themed release PR names, or
  calendar release branches as default maintenance.
- Prefer small issue/PR-scoped planning and package version releases driven by
  user-facing changes.
- Use GitHub milestones only for explicit, short-lived initiatives with an
  active owner; they are optional scheduling aids, not release gates.
- Name release branches and release PRs for the version or concrete change set
  they contain.
- When publishing, update package metadata, docs, examples, and GitHub
  Releases/PyPI notes as needed. Do not add README callouts that point to
  monthly milestones.

## Keep This File Up To Date

Update this file whenever the contributor workflow changes, especially when
setup commands, validation commands, or the public API expectations change.
