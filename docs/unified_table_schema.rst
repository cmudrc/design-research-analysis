Unified Table Schema
====================

Purpose
-------

The unified table schema is the canonical input contract across all analysis
families in this package. It enables repeatable pipelines while still allowing
loose real-world data.

If your input originated in ``design-research-experiments``, see
:doc:`experiments_handoff` for the recommended ``events.csv`` validation and
join workflow.

Column Expectations
-------------------

Required:

- ``timestamp``

Strongly recommended:

- ``record_id``
- ``text``
- ``session_id``
- ``actor_id``
- ``event_type``

Optional:

- ``meta_json``

Derived in the common experiments handoff when needed:

- ``record_id``
- ``actor_id``
- ``event_type``

Loose Schema Strategy
---------------------

Missing values for ``actor_id`` and ``event_type`` can be derived with
deterministic mapper functions before running sequence analyses.

In the experiments export handoff, ``record_id`` may also be derived when the
upstream artifact keeps stable event rows but does not emit explicit record
identifiers.

Use :doc:`experiments_handoff` when the input came from
``design-research-experiments`` and you want the concrete ``events.csv`` ->
validation -> downstream-analysis path rather than the abstract schema view.

Key API surfaces:

- :func:`design_research_analysis.coerce_unified_table`
- :func:`design_research_analysis.validate_unified_table`
- :func:`design_research_analysis.derive_columns`
- ``design_research_analysis.table.group_rows``

Example
-------

.. code-block:: python

   from design_research_analysis import (
       derive_columns,
       validate_unified_table,
   )

   rows = [
       {"timestamp": "2026-01-01T10:00:00Z", "text": "hello", "speaker": "alice"},
       {"timestamp": "2026-01-01T10:00:01Z", "text": "world", "speaker": "bob"},
   ]

   rows = derive_columns(
       rows,
       actor_mapper=lambda row: row["speaker"],
       event_mapper=lambda _row: "utterance",
   )
   report = validate_unified_table(rows)
   assert report.is_valid
