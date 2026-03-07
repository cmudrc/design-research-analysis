Installation
============

Package Install
---------------

.. code-block:: bash

   pip install design-research-analysis

Editable Install
----------------

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-analysis.git
   cd design-research-analysis
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e ".[dev]"

Maintainer Shortcut
-------------------

.. code-block:: bash

   make dev

Analysis Extras
---------------

Install extras by analysis family.

.. code-block:: bash

   pip install -e ".[seq]"
   pip install -e ".[lang,embeddings]"
   pip install -e ".[stats,data]"
   pip install -e ".[all]"

Use :doc:`dependencies_and_extras` for family-level guidance and tradeoffs.
