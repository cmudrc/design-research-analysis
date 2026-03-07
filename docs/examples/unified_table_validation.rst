Unified Table Validation
========================

Source: ``examples/unified_table_validation.py``

Introduction
------------

Show how lightly structured transcript rows can be normalized into the unified
table contract with deterministic mapper functions.

Technical Implementation
------------------------

1. Start from rows with ``speaker`` instead of canonical actor fields.
2. Derive ``actor_id`` and ``event_type`` using mapper callbacks.
3. Validate the resulting rows against the unified-table contract.

.. literalinclude:: ../../examples/unified_table_validation.py
   :language: python
   :lines: 20-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/unified_table_validation.py

Prints a validation report dictionary with required/recommended field checks and
warnings for missing recommended columns.

References
----------

- docs/unified_table_schema.rst
