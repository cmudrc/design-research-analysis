Sequence From Table
===================

Source: ``examples/sequence_from_table.py``

Introduction
------------

Convert a short event log into a transition model that describes the team's
design-process flow.

Technical Implementation
------------------------

1. Construct event rows with timestamps, session IDs, and event labels.
2. Fit a first-order Markov chain with additive smoothing.
3. Print state order and transition probabilities.

.. literalinclude:: ../../examples/sequence_from_table.py
   :language: python
   :lines: 20-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/sequence_from_table.py

Prints a tuple state list and a dense transition matrix suitable for downstream
visualization or comparison across sessions.

References
----------

- docs/workflows.rst
