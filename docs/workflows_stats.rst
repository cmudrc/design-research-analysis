Statistical Workflows
=====================

Use statistical workflows when inferential claims, effect estimation, and model
fit quality are central outputs.

Typical Questions
-----------------

- Are condition-level effects statistically distinguishable?
- What is the estimated effect size and uncertainty?
- How do covariates and hierarchical structure influence outcomes?

Key API Entry Points
--------------------

- :func:`design_research_analysis.compare_groups`
- :func:`design_research_analysis.build_condition_metric_table`
- :func:`design_research_analysis.compare_condition_pairs`
- :func:`design_research_analysis.fit_regression`
- :func:`design_research_analysis.fit_mixed_effects`
- :func:`design_research_analysis.bootstrap_ci`
- :func:`design_research_analysis.permutation_test`

CLI Path
--------

.. code-block:: bash

   design-research-analysis run-stats \
     --input data/events.csv \
     --summary-json artifacts/stats.json \
     --mode regression \
     --x-columns x1,x2 \
     --y-column y
