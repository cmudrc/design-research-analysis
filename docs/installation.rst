Installation
============

Package Install
---------------

.. code-block:: bash

   python -m pip install --upgrade pip
   pip install design-research-analysis

Editable Install
----------------

Use editable installs when contributing from a local checkout.

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

   pip install "design-research-analysis[seq]"
   pip install "design-research-analysis[lang,embeddings]"
   pip install "design-research-analysis[maps]"
   pip install "design-research-analysis[stats,data]"
   pip install "design-research-analysis[all]"

When working from a source checkout, replace ``design-research-analysis`` with ``.``
to install the same extras in editable mode.

Use :doc:`dependencies_and_extras` for family-level guidance and tradeoffs.
