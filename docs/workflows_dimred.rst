Dimensionality-Reduction Workflows
==================================

Use dimred workflows when embedding structure must be inspected, compared, or
visualized.

Typical Questions
-----------------

- Do records cluster by condition, role, or phase?
- Are semantic spaces separable across treatments?
- Which projection best preserves interpretable structure?

Key API Entry Points
--------------------

- :func:`design_research_analysis.embed_records`
- :func:`design_research_analysis.reduce_dimensions`
- :func:`design_research_analysis.cluster_projection`
- :func:`design_research_analysis.compute_design_space_coverage`
- :func:`design_research_analysis.compute_idea_space_trajectory`
- :func:`design_research_analysis.compute_divergence_convergence`

Projection-Space Diagnostics
----------------------------

Coverage and trajectory metrics can be computed on raw embeddings or on a
reduced projection. Prefer projection-space metrics when you need lightweight,
human-readable summaries that match plotted coordinates and CLI exports. Prefer
raw embedding-space metrics when preserving the full geometry matters more than
plot interpretability.

CLI Path
--------

.. code-block:: bash

   design-research-analysis run-dimred \
     --input data/events.csv \
     --summary-json artifacts/dimred.json \
     --projection-csv artifacts/projection.csv

The dimred summary JSON includes projection-space clustering, coverage, and
trajectory diagnostics. When ``session_id`` and ``timestamp`` are present, the
trajectory block is grouped and ordered from those unified-table columns.
