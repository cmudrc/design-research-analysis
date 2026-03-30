# Documentation Maintenance

## Build Docs Locally

- `make docs-check`
- `make docs-build`

## Example Page Generation

Example pages are generated from runnable scripts via `scripts/generate_example_docs.py`.
Update example docstrings/comments and rerun docs checks after changes.

## Docstring Style

Use Google-style docstrings where policy applies.
Run `make docstrings-check` before merge.

## Page-Writing Conventions

- Keep the homepage short: title, tagline, concise framing, quickstart callout, section-oriented links, and only the minimum ecosystem/contribution notes needed for orientation.
- Keep the root hidden home-page toctree section-first so the PyData header and sidebar stay stable.
- Emphasize unified-table contracts, reproducibility, and empirical interpretation.
- Keep snippets runnable and aligned with public API exports.

## Table vs Prose Rule

Prefer compact tables for scanning. Preserve nuance in narrative paragraphs directly below the table. Do not use tables to carry long explanatory sentences.

## Cross-links

Use `:doc:` for internal links and explicitly connect analysis outputs to experiments/agents/problems contexts when relevant.

## Branding

- The ecosystem figure is the source of truth for package colors.
- This repo's canonical docs brand color is `#4D8687`.
- Keep docs CSS tokens, `drc-light.png`, `drc-dark.png`, and `favicon.ico` aligned when updating docs styling.

## API Page Updates

When public exports change, update:

- `docs/api.rst`
- workflow/CLI references
- quickstart/examples snippets
