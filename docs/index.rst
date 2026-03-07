design-research-analysis
========================

A typed, reusable analysis library for recurring design-research event tables.

Use it to:

- validate and normalize unified event-table inputs,
- run sequence, language, dimensionality-reduction, and statistics workflows, and
- capture reproducibility metadata for repeatable downstream reporting.

Highlights
----------

- Unified-table contracts with deterministic mapper-based column derivation.
- Sequence analysis over event rows (Markov chains and optional HMM workflows).
- Language analysis helpers for semantic convergence, trajectories, topics, and sentiment.
- Embedding and projection pipelines with clustering support.
- Statistical wrappers spanning group tests, regression, mixed effects, and power analysis.
- Dataset profiling, schema validation, and codebook helpers.
- CLI commands that write structured JSON summaries for pipeline integration.

Typical Workflow
----------------

1. Normalize and validate unified-table records.
2. Run one or more analysis families (sequence, language, dimred, stats).
3. Persist JSON summaries and optional artifact outputs.
4. Attach runtime provenance for reproducible reports.

Start Here
----------

- :doc:`quickstart` for a compact end-to-end setup path.
- :doc:`examples/index` for example-by-example pages generated from runnable script docstrings.
- :doc:`workflows` for common analysis patterns and API entry points.
- :doc:`unified_table_schema` for the canonical event-table contract.
- :doc:`cli_reference` for command-by-command behavior and options.
- :doc:`analysis_recipes` for copy/paste recipe snippets.
- :doc:`dependencies_and_extras` for install profiles and release checks.
- :doc:`api` for the supported top-level public surface.
- `CONTRIBUTING.md <https://github.com/cmudrc/design-research-analysis/blob/main/CONTRIBUTING.md>`_
  for contribution workflow and quality gates.

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   quickstart
   examples/index
   workflows
   unified_table_schema
   cli_reference
   analysis_recipes
   dependencies_and_extras

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api
