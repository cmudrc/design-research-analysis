Basic Usage
===========

Source: ``examples/basic_usage.py``

Introduction
------------

Use a tiny unified table from one team session to run the most common baseline:
schema validation, event-sequence transitions, and language convergence.

Technical Implementation
------------------------

1. Validate required and recommended unified-table columns.
2. Fit a first-order Markov chain from event transitions.
3. Compare the fitted Markov chain to a small alternate session trace.
4. Compute semantic convergence using a deterministic custom embedder.

.. literalinclude:: ../../examples/basic_usage.py
   :language: python
   :lines: 21-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/basic_usage.py

Prints the ordered Markov states, transition matrix, one model-comparison summary,
and one convergence label for ``team-a``.

References
----------

- docs/workflows.rst
