Analysis Recipes
================

Sequence Recipe
---------------

.. code-block:: python

   from design_research_analysis import fit_markov_chain_from_table

   result = fit_markov_chain_from_table(rows, order=1, smoothing=1.0)
   print(result.transition_matrix)

Model Comparison Recipe
-----------------------

.. code-block:: python

   from design_research_analysis import fit_markov_chain_from_table

   baseline = fit_markov_chain_from_table(rows_a, order=1, smoothing=1.0)
   variant = fit_markov_chain_from_table(rows_b, order=1, smoothing=1.0)

   difference = baseline - variant
   effect = baseline / variant
   print(difference.to_dict())
   print(effect.effect_size)

Language Recipe (Custom Embedder)
---------------------------------

.. code-block:: python

   from design_research_analysis import compute_language_convergence

   lookup = {
       "alpha": [2.0, 0.0],
       "beta": [1.0, 0.0],
       "gamma": [0.0, 0.0],
   }

   convergence = compute_language_convergence(
       rows,
       window_size=1,
       embedder=lambda texts: [lookup[text] for text in texts],
   )
   print(convergence.direction_by_group)

Dimensionality Reduction Recipe
-------------------------------

.. code-block:: python

   import numpy as np
   from design_research_analysis import (
       cluster_projection,
       compute_design_space_coverage,
       compute_divergence_convergence,
       compute_idea_space_trajectory,
       reduce_dimensions,
   )

   vectors = np.asarray(
       [
           [1.0, 0.0, 0.1],
           [0.9, 0.2, 0.1],
           [0.0, 1.0, 0.2],
           [0.1, 0.9, 0.3],
       ]
   )
   projection = reduce_dimensions(vectors, method="pca", n_components=2)
   clusters = cluster_projection(projection.projection, n_clusters=2)
   coverage = compute_design_space_coverage(projection)
   trajectory = compute_idea_space_trajectory(
       projection,
       timestamps=[0, 1, 2, 3],
       groups=["session-a"] * 4,
   )
   dynamics = compute_divergence_convergence(trajectory, window=2)
   print(clusters["labels"])
   print(coverage["pairwise_spread"]["mean"])
   print(dynamics["groups"]["session-a"]["dominant_direction"])

Statistics Recipe
-----------------

.. code-block:: python

   from design_research_analysis import fit_regression

   X = [[0.0], [1.0], [2.0], [3.0]]
   y = [1.0, 3.0, 5.0, 7.0]
   reg = fit_regression(X, y, feature_names=["x"])
   print(reg.coefficients, reg.intercept, reg.r2)

Experiment Condition Comparison Recipe
--------------------------------------

.. code-block:: python

   from design_research_analysis import (
       build_condition_metric_table,
       compare_condition_pairs,
   )

   condition_metric_rows = build_condition_metric_table(
       runs_rows,
       metric="market_share_proxy",
       condition_column="selection_strategy",
       conditions=conditions_rows,
       evaluations=evaluations_rows,
   )

   report = compare_condition_pairs(
       condition_metric_rows,
       condition_pairs=[
           ("profit_focus_prompt", "neutral_prompt"),
           ("neutral_prompt", "random_selection"),
       ],
       alternative="greater",
       seed=17,
   )
   print(report.render_brief())
   print(report.to_significance_rows())

Dataset And Provenance Recipe
-----------------------------

.. code-block:: python

   import pandas as pd
   from design_research_analysis import (
       capture_run_context,
       generate_codebook,
       profile_dataframe,
       validate_dataframe,
       write_run_manifest,
   )

   frame = pd.DataFrame({"participant_id": [1, 2, 3], "condition": ["A", "A", "B"]})
   profile = profile_dataframe(frame)
   validation = validate_dataframe(frame, {"participant_id": {"unique": True, "nullable": False}})
   codebook = generate_codebook(frame)
   context = capture_run_context(seed=7)
   write_run_manifest(context, "artifacts/run_manifest.json")
   print(profile["n_rows"], validation["ok"], list(codebook["column"]))
