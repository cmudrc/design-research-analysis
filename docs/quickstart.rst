Quickstart
==========

This example shows the shortest meaningful path through
``design-research-analysis``.

1. Install
----------

.. code-block:: bash

   pip install design-research-analysis

Or install from source:

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-analysis.git
   cd design-research-analysis
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e .

2. Minimal Runnable Example
---------------------------

.. code-block:: python

   from design_research_analysis import (
       coerce_unified_table,
       fit_markov_chain_from_table,
       validate_unified_table,
   )

   rows = coerce_unified_table(
       [
           {"timestamp": "2026-01-01T00:00:00", "session_id": "s1", "actor_id": "a", "event_type": "ideate"},
           {"timestamp": "2026-01-01T00:00:05", "session_id": "s1", "actor_id": "a", "event_type": "refine"},
           {"timestamp": "2026-01-01T00:00:10", "session_id": "s1", "actor_id": "a", "event_type": "evaluate"},
       ]
   )

   report = validate_unified_table(rows)
   if not report.is_valid:
       raise RuntimeError("; ".join(report.errors))

   result = fit_markov_chain_from_table(rows)
   print(result.states)

3. What Happened
----------------

You validated a unified event table and fit a sequence model from those records.
This demonstrates the package design: schema discipline first, analysis second.

4. Where To Go Next
-------------------

- :doc:`concepts`
- :doc:`experiments_handoff`
- :doc:`typical_workflow`
- :doc:`workflows`
- :doc:`examples/index`

Ecosystem Note
--------------

In a typical study, ``design-research-agents`` provides executable
participants, ``design-research-problems`` supplies the task,
``design-research-experiments`` defines the study structure, and
``design-research-analysis`` interprets the resulting records.

If you are starting from exported study artifacts rather than an in-memory
table, use :doc:`experiments_handoff` for the recommended ``events.csv`` to
analysis path.
