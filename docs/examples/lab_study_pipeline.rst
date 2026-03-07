Lab Study Pipeline
==================

Source: ``examples/lab_study_pipeline.py``

Introduction
------------

Run a realistic small-sample lab workflow across control vs reframed conditions
with sequence, language, dataset, dimensionality-reduction, statistical, and
provenance outputs.

Technical Implementation
------------------------

1. Build an in-memory unified event table with condition labels and outcome fields.
2. Validate table quality and compute sequence/language summaries.
3. Profile and validate a dataframe schema; generate a codebook.
4. Run group comparison, regression, bootstrap, permutation, and power helpers.
5. Run PCA + clustering and write a reproducibility manifest with attached
   provenance payload.

.. literalinclude:: ../../examples/lab_study_pipeline.py
   :language: python
   :lines: 26-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/lab_study_pipeline.py

Prints concise summaries for state count, convergence labels, sentiment totals,
schema/profile diagnostics, key statistical metrics, clustering labels, and the
manifest path under ``artifacts/runtime``.

References
----------

- docs/workflows.rst
- docs/analysis_recipes.rst
