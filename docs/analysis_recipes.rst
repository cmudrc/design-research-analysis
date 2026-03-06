Analysis Recipes
================

Sequence Recipe
---------------

.. code-block:: python

   from design_research_analysis import fit_markov_chain_from_table

   result = fit_markov_chain_from_table(rows, order=1, smoothing=1.0)
   print(result.transition_matrix)

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
   from design_research_analysis import cluster_projection, reduce_dimensions

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
   print(clusters["labels"])

Statistics Recipe
-----------------

.. code-block:: python

   from design_research_analysis import fit_regression

   X = [[0.0], [1.0], [2.0], [3.0]]
   y = [1.0, 3.0, 5.0, 7.0]
   reg = fit_regression(X, y, feature_names=["x"])
   print(reg.coefficients, reg.intercept, reg.r2)
