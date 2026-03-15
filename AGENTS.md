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
- The reproducible interpreter target lives in `.python-version` (`3.12.12`).
- Install local tooling with `make dev`.
- For a frozen environment based on `uv.lock`, use `make repro`.

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
  - `make release-check`

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

## Release Naming

- Theme: Pittsburgh / CMU people, places, and institutions.
- Monthly release names are shared across milestone titles, release PR titles,
  and release branches.
  - Milestone title / PR title: `{base name} - {Month YYYY}`
  - Release branch: slugified full title, for example
    `mellon-metrics-may-2026`
- Milestone descriptions must use:
  - `Tracks {previous month YYYY} work.`
  - `Theme source: <url>`
- Release PR bodies must repeat the same `Theme source:` link used on the
  milestone.
- Never reuse an exact base name or the same primary subject across any month
  or any of the four design-research module repos unless all four `AGENTS.md`
  files are intentionally updated together.
- Before adding a new release name, check the `Release Naming` tables in all
  four repos to avoid repeats.

| Due date | Base name | Source subject |
| --- | --- | --- |
| April 1, 2026 | Allegheny Analysis | Allegheny River |
| May 1, 2026 | Mellon Metrics | Andrew Mellon |
| June 1, 2026 | Carnegie Calculus | Andrew Carnegie |
| July 1, 2026 | Resnik Readout | Judith Resnik |
| August 1, 2026 | Cathedral Calculus | Cathedral of Learning |
| September 1, 2026 | Schenley Signals | Schenley Park |
| October 1, 2026 | Oakland Observations | Oakland, Pittsburgh |
| November 1, 2026 | Nationality Notes | Nationality Rooms |
| December 1, 2026 | Doherty Diagnostics | Doherty Hall |
| January 1, 2027 | Tartan Trends | Carnegie Mellon Tartans |

## Keep This File Up To Date

Update this file whenever the contributor workflow changes, especially when
setup commands, validation commands, or the public API expectations change.
