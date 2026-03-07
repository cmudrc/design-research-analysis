Dimred PCA
==========

Source: ``examples/dimred_pca.py``

Introduction
------------

Reduce higher-dimensional study vectors to two dimensions for plotting and then
cluster projected points into interpretable groups.

Technical Implementation
------------------------

1. Build a deterministic in-memory numeric matrix.
2. Run PCA projection with two components.
3. Run k-means clustering on projected coordinates.

.. literalinclude:: ../../examples/dimred_pca.py
   :language: python
   :lines: 20-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/dimred_pca.py

Prints a projection dictionary (shape, method, variance ratio) and a clustering
dictionary with labels, centers, and config.

References
----------

- docs/analysis_recipes.rst
