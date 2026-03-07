# Documentation Maintenance

## Build Docs Locally

- `make docs-check`
- `make docs`

## Example Page Generation

Example pages are generated from runnable scripts via `scripts/generate_example_docs.py`.
Update example docstrings/comments and rerun docs checks after changes.

## Docstring Style

Use Google-style docstrings where policy applies.
Run `make docstrings-check` before merge.

## Page-Writing Conventions

- Keep homepages in this order: title, tagline, what it does, highlights, typical workflow, ecosystem integration, start here.
- Emphasize unified-table contracts, reproducibility, and empirical interpretation.
- Keep snippets runnable and aligned with public API exports.

## Table vs Prose Rule

Prefer compact tables for scanning. Preserve nuance in narrative paragraphs directly below the table. Do not use tables to carry long explanatory sentences.

## Cross-links

Use `:doc:` for internal links and explicitly connect analysis outputs to experiments/agents/problems contexts when relevant.

## API Page Updates

When public exports change, update:

- `docs/api.rst`
- workflow/CLI references
- quickstart/examples snippets
