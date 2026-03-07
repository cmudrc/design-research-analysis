Language Custom Embedder
========================

Source: ``examples/language_custom_embedder.py``

Introduction
------------

Analyze one session's discourse shift without external model dependencies by
supplying a local deterministic embedding lookup.

Technical Implementation
------------------------

1. Create three timestamped utterances in one session.
2. Map each utterance to a fixed numeric vector with ``embedder=...``.
3. Compute convergence trajectories and slope-based direction labels.

.. literalinclude:: ../../examples/language_custom_embedder.py
   :language: python
   :lines: 20-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/language_custom_embedder.py

Prints a serialized convergence result containing trajectories, per-group slopes,
and direction labels for ``team-b``.

References
----------

- docs/analysis_recipes.rst
