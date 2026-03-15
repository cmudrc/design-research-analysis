Concepts
========

Unified Event Tables
--------------------

The primary input is a unified event-table representation for design-study
records. This contract enables analysis functions to compose across data
sources.

Required and Recommended Columns
--------------------------------

Required columns are intentionally minimal. Recommended columns preserve
interpretability for actor-level, session-level, and event-type analyses.

Derivation and Validation
-------------------------

Real research datasets are often incomplete or heterogenous. Validation and
mapper-based derivation formalize how missing analytical fields are resolved,
which improves reproducibility and reduces undocumented data wrangling.

Analysis Families
-----------------

- sequence: transition and temporal-structure questions
- language: semantic convergence and discourse questions
- embedding maps: embedding structure, trajectories, and scalar-overlay questions
- stats: inferential and effect-size questions

Artifacts and Summaries
-----------------------

The package writes machine-readable summaries and optional tables/plots so
analyses can be incorporated into larger experimental pipelines.

Provenance and Repeatability
----------------------------

Runtime context capture and manifest writing support reruns, audit trails, and
transparent reporting of environment assumptions.
