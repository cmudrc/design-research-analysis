Idea Space Metrics
==================

Source: ``examples/idea_space_metrics.py``

Introduction
------------

Build a compact idea-space analysis from deterministic vectors, then render the
timeline, trajectory, and convergence views that support interpretation.

Technical Implementation
------------------------

1. Construct a single-session unified table plus a deterministic numeric space.
2. Compute projection-space coverage, trajectory summaries, and divergence markers.
3. Render three plots and save them to a stable output directory.

.. literalinclude:: ../../examples/idea_space_metrics.py
   :language: python
   :lines: 21-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/idea_space_metrics.py

Prints the convex-hull support flag, the dominant trajectory direction, and the
directory containing the generated PNG figures.

References
----------

- docs/workflows_dimred.rst
- docs/workflows_sequence.rst
