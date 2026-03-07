Quickstart
==========

Requires Python 3.12+ and assumes you are working from the repository root.

Create and activate a virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

Path A: Python API
------------------

Use this when you want analysis setup and execution in notebooks or scripts.

1. Install development tooling and package dependencies:

.. code-block:: bash

   make dev

2. Run a compact API walkthrough example:

.. code-block:: bash

   PYTHONPATH=src python examples/basic_usage.py

3. Explore additional runnable examples in ``examples/README.md``.

Path B: CLI
-----------

Use this when you want file-driven workflows and machine-readable artifacts.

1. Validate a unified table:

.. code-block:: bash

   design-research-analysis validate-table \
     --input data/events.csv \
     --summary-json artifacts/validate.json

2. Run a sequence baseline:

.. code-block:: bash

   design-research-analysis run-sequence \
     --input data/events.csv \
     --summary-json artifacts/sequence.json \
     --mode markov

3. Run language and export trajectory output:

.. code-block:: bash

   design-research-analysis run-language \
     --input data/events.csv \
     --summary-json artifacts/language.json \
     --trajectory-csv artifacts/language_trajectory.csv

Checks and Docs
---------------

.. code-block:: bash

   make test
   make docs-check
   make docs-build

Next Steps
----------

- Install profiles and release maintenance checks: :doc:`dependencies_and_extras`
- Unified table contract and mapper strategy: :doc:`unified_table_schema`
- Command reference for all subcommands: :doc:`cli_reference`
- Supported top-level exports: :doc:`api`
