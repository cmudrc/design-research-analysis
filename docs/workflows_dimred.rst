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

CLI Path
--------

.. code-block:: bash

   design-research-analysis run-dimred \
     --input data/events.csv \
     --summary-json artifacts/dimred.json \
     --projection-csv artifacts/projection.csv
