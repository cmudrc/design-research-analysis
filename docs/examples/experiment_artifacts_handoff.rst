Experiment Artifacts Handoff
============================

Source: ``examples/experiment_artifacts_handoff.py``

Introduction
------------

Start from a study-output directory shaped like a
``design-research-experiments`` export and run standard analysis workflows
without manually joining ``runs.csv``, ``conditions.csv``, ``events.csv``, or
``evaluations.csv``.

Technical Implementation
------------------------

1. Write a tiny deterministic artifact bundle that stands in for an exported
   experiment.
2. Validate the bundle through top-level artifact helpers.
3. Run condition comparisons, Markov-chain comparisons, and regression directly
   from the artifact directory.

.. literalinclude:: ../../examples/experiment_artifacts_handoff.py
   :language: python
   :lines: 24-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/experiment_artifacts_handoff.py

Prints validation status, derived table sizes, condition-comparison count,
Markov-chain labels, transition-comparison estimate, and regression coefficients.

References
----------

- docs/experiments_handoff.rst
