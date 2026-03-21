Typical Workflow
================

1. Choose inputs
----------------

Load event-table records and identify the study context (conditions, actors,
outcomes).

If you are working from ``design-research-experiments`` exports, start with
``events.csv`` and use :doc:`experiments_handoff` for the canonical validation
and rejoin flow back to ``runs.csv`` and ``evaluations.csv``.

2. Instantiate core objects
---------------------------

Create unified-table validation/derivation config and choose analysis-family
entry points.

3. Execute or inspect
---------------------

Run sequence, language, embedding-map, and/or statistical workflows.

4. Capture artifacts
--------------------

Write structured summaries, optional exports, and run-manifest metadata.

5. Connect to the next library
------------------------------

Feed interpreted findings back to ``design-research-experiments`` for protocol
refinement and to agent/problem selection decisions for future studies.

Choosing Analysis Families
--------------------------

Sequence analysis is best when order and transitions matter. Language analysis
is best when semantic change or discourse properties matter. Embedding maps are
best when embedding geometry, trajectories, and clustering structure matter.
Statistical workflows are best when inferential claims and effect-size
estimation matter.
