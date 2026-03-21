Workflows
=========

Analysis families map to different empirical questions.

.. list-table::
   :header-rows: 1

   * - Family
     - Use when
   * - Sequence
     - Order and transitions matter
   * - Language
     - Text content and convergence matter
   * - Embedding maps
     - Embedding structure, trajectories, or scalar overlays need inspection
   * - Statistics
     - Effect-size estimation or inference matters

Sequence workflows are centered on transition structure and temporal dynamics.
Language workflows are centered on discourse and semantic movement. Embedding
map workflows are centered on structure discovery and trajectory interpretation
in learned representations. Statistical workflows are centered on hypothesis
testing, regression, and uncertainty quantification.

Subpages
--------

.. toctree::
   :maxdepth: 1

   workflows_sequence
   workflows_language
   workflows_embedding_maps
   workflows_stats

CLI Integration
---------------

The CLI mirrors these families with deterministic JSON summary outputs. Use
``validate-table`` before analysis-family runs when ingesting new datasets, then
persist ``--summary-json`` outputs as pipeline artifacts for reproducible
reporting.
