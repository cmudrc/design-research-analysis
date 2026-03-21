Condition Pair Significance
===========================

Source: ``examples/condition_pair_significance.py``

Introduction
------------

Turn canonical ``runs.csv`` / ``conditions.csv`` / ``evaluations.csv`` style
rows into a joined condition-metric table, then compute pairwise permutation
tests and effect sizes without hand-rolled analysis glue.

Technical Implementation
------------------------

1. Define in-memory canonical export rows for runs, conditions, and evaluations.
2. Build a normalized run-level table for ``market_share_proxy`` by joining
   condition labels onto evaluation metrics.
3. Compare ordered condition pairs and print a concise brief plus one
   significance row shaped for ``design_research_experiments.render_significance_brief``.

.. literalinclude:: ../../examples/condition_pair_significance.py
   :language: python
   :lines: 24-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/condition_pair_significance.py

Prints the normalized joined-row count, a markdown-ready condition comparison
brief, and the first structured significance row.

References
----------

- docs/experiments_handoff.rst
- docs/analysis_recipes.rst
