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
- :func:`design_research_analysis.compute_interrater_reliability`
- :func:`design_research_analysis.fit_regression`
- :func:`design_research_analysis.fit_mixed_effects`
- :func:`design_research_analysis.bootstrap_ci`
- :func:`design_research_analysis.permutation_test`

Inter-Rater Reliability
-----------------------

Pass an item-by-rater matrix to the nominal reliability helper. ``None`` and
``NaN`` are treated as missing. Cohen's and Fleiss' kappa use complete items;
Krippendorff's alpha retains items with at least two observed ratings.

.. code-block:: python

   import design_research_analysis as dran

   codings = [
       ["problem", "problem", "problem"],
       ["solution", "solution", "problem"],
       ["evaluation", "evaluation", "evaluation"],
       ["solution", "solution", "solution"],
   ]
   result = dran.compute_interrater_reliability(
       codings,
       method="krippendorff_alpha",
       n_bootstrap=500,
       seed=17,
   )
   print(result.coefficient, result.confidence_interval)

Supported methods are ``cohen_kappa`` for exactly two raters,
``fleiss_kappa`` for two or more raters, and nominal
``krippendorff_alpha``. Bootstrap intervals resample items and report how many
non-degenerate resamples contributed to the interval.

CLI Path
--------

.. code-block:: bash

   design-research-analysis run-stats \
     --input data/events.csv \
     --summary-json artifacts/stats.json \
     --mode regression \
     --x-columns x1,x2 \
     --y-column y
