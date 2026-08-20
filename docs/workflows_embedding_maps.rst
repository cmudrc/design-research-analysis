Embedding Maps Workflows
========================

Use embedding-map workflows when embedding structure, trajectories, or scalar
value overlays must be inspected, compared, or visualized.

Install ``design-research-analysis[maps,embeddings]`` when records begin as
text and the package should create sentence embeddings. Install ``[maps]``
alone when you already have numeric vectors or use the CLI's
``--feature-columns`` path. PCA projection and plotting with a custom embedder
are available from the base install; ``maps`` supplies the additional manifold
and clustering backends.

Typical Questions
-----------------

- Do records cluster by condition, role, or phase?
- Are semantic spaces separable across treatments?
- Which map best preserves interpretable structure and trajectory legibility?
- How do traces move through the same map as value signals change?

Key API Entry Points
--------------------

- :func:`design_research_analysis.embed_records`
- :func:`design_research_analysis.build_embedding_map`
- :func:`design_research_analysis.compare_embedding_maps`
- :func:`design_research_analysis.cluster_embedding_map`
- :func:`design_research_analysis.compute_design_space_coverage`
- :func:`design_research_analysis.compute_idea_space_trajectory`
- :func:`design_research_analysis.compute_divergence_convergence`
- :func:`design_research_analysis.plot_embedding_map`
- :func:`design_research_analysis.plot_embedding_map_grid`

Map-Space Diagnostics
---------------------

Coverage and trajectory metrics can be computed on raw embeddings or on a
lower-dimensional embedding map. Prefer map-space metrics when you want
summaries that line up directly with plotted coordinates and CLI exports.
Prefer raw embedding-space metrics when preserving the full geometry matters
more than human-readable visuals.

CLI Path
--------

.. code-block:: bash

   design-research-analysis run-embedding-maps \
     --input data/events.csv \
     --summary-json artifacts/embedding_maps.json \
     --map-csv artifacts/embedding_maps.csv \
     --method pca \
     --method umap \
     --trace-column session_id \
     --order-column timestamp \
     --comparison-png artifacts/embedding_maps.png

The embedding-maps summary JSON includes per-method clustering, coverage, and
trajectory diagnostics. When trace and order columns are supplied, trajectories
follow those fields; otherwise the CLI falls back to ``session_id`` and
``timestamp`` when present.

Because the command embeds the ``text`` column when ``--feature-columns`` is
omitted, the command shown above requires both ``maps`` and ``embeddings``.
