Quickstart
==========

This package targets Python 3.12+ and uses a standard ``src/`` layout.

Local development setup:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   make dev
   make test

Run all bundled examples:

.. code-block:: bash

   make run-examples

Install optional analysis extras as needed:

.. code-block:: bash

   pip install -e ".[seq]"
   pip install -e ".[embeddings]"
   pip install -e ".[lang]"
   pip install -e ".[dimred]"
   pip install -e ".[stats]"

Run a CLI check:

.. code-block:: bash

   design-research-analysis validate-table \
     --input data/events.csv \
     --summary-json artifacts/validate.json

Build the docs:

.. code-block:: bash

   make docs
