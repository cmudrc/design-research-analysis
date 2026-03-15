Embedding Maps Workflows
========================

Use embedding-map workflows when embedding structure, trajectories, or scalar
value overlays must be inspected, compared, or visualized.

Typical Questions
-----------------

- Do records cluster by condition, role, or phase?
- Are semantic spaces separable across treatments?
- Which map best preserves interpretable structure and trajectory legibility?

Key API Entry Points
--------------------

- :func:`design_research_analysis.embed_records`
- :func:`design_research_analysis.build_embedding_map`
- :func:`design_research_analysis.compare_embedding_maps`
- :func:`design_research_analysis.plot_embedding_map`
- :func:`design_research_analysis.plot_embedding_map_grid`
- :func:`design_research_analysis.cluster_embedding_map`

CLI Path
--------

.. code-block:: bash

   design-research-analysis run-embedding-maps \
     --input data/events.csv \
     --summary-json artifacts/embedding_maps.json \
     --map-csv artifacts/embedding_maps.csv
