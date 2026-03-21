Embedding Maps Trajectories
===========================

Source: ``examples/embedding_maps_trajectories.py``

Introduction
------------

Map ordered design traces into a shared coordinate space, then overlay their
paths and scalar values so exploration behavior is visible instead of just the
point cloud.

Technical Implementation
------------------------

1. Build a tiny unified-table-like dataset with record IDs, trace IDs, steps,
   text, and performance values.
2. Embed the text deterministically with a local lookup and build a PCA
   embedding map.
3. Render a single value-colored trajectory map and a comparison grid; when no
   optional mapping backend is installed, compare PCA with a reflected PCA map
   so the plotting workflow still runs end to end.

.. literalinclude:: ../../examples/embedding_maps_trajectories.py
   :language: python
   :lines: 26-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/embedding_maps_trajectories.py

Writes ``artifacts/examples/embedding_map.png`` and
``artifacts/examples/embedding_map_grid.png``, then prints the method names
included in the comparison.

References
----------

- docs/workflows_embedding_maps.rst
