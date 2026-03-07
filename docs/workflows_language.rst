Language Workflows
==================

Use language workflows when textual content and semantic change are central
study signals.

Typical Questions
-----------------

- Are participants converging semantically over time?
- Do groups diverge in topical framing?
- How does sentiment shift across phases?

Key API Entry Points
--------------------

- :func:`design_research_analysis.compute_language_convergence`
- :func:`design_research_analysis.compute_semantic_distance_trajectory`
- :func:`design_research_analysis.fit_topic_model`
- :func:`design_research_analysis.score_sentiment`

CLI Path
--------

.. code-block:: bash

   design-research-analysis run-language \
     --input data/events.csv \
     --summary-json artifacts/language.json \
     --trajectory-csv artifacts/language_trajectory.csv
