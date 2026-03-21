Mechanical Design Review Analysis
=================================

Source: ``examples/mechanical_design_review_analysis.py``

Introduction
------------

Analyze a compact mechanical design review centered on a lightweight mounting
bracket redesign. The example keeps the workflow small while surfacing the
sequence, statistics, embedding, and reporting helpers that are useful in
engineering design studies.

Technical Implementation
------------------------

1. Validate and summarize a tiny unified event table from two bracket-review sessions.
2. Fit and compare Markov-chain traces, build a condition metric table, and run
   a permutation-style pair comparison on mass-oriented scores.
3. Embed the review notes with a deterministic custom embedder and render quick
   transition visualizations for the session traces.

.. literalinclude:: ../../examples/mechanical_design_review_analysis.py
   :language: python
   :lines: 21-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/mechanical_design_review_analysis.py

The script prints validation status, state labels, one significance summary, and
an embedding shape for the bracket-design review sessions.
