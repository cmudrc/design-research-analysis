Stats Regression
================

Source: ``examples/stats_regression.py``

Introduction
------------

Estimate a linear trend between prototype iteration index and novelty score for a
compact design-study sample.

Technical Implementation
------------------------

1. Define one explanatory feature (iteration count).
2. Fit an OLS regression through the package helper.
3. Print the serialized coefficient and fit diagnostics.

.. literalinclude:: ../../examples/stats_regression.py
   :language: python
   :lines: 19-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/stats_regression.py

Prints regression coefficients, intercept, R2, MSE, and input-shape metadata.

References
----------

- docs/analysis_recipes.rst
