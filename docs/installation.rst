Installation
============

Package Install
---------------

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install design-research-analysis

Editable Install
----------------

Use editable installs when contributing from a local checkout.

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-analysis.git
   cd design-research-analysis
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"

Maintainer Shortcut
-------------------

.. code-block:: bash

   make dev

Analysis Extras
---------------

Install extras by analysis family.

.. code-block:: bash

   python -m pip install "design-research-analysis[seq]"
   python -m pip install "design-research-analysis[lang,embeddings]"
   python -m pip install "design-research-analysis[maps,embeddings]"
   python -m pip install "design-research-analysis[maps]"
   python -m pip install "design-research-analysis[stats,data]"
   python -m pip install "design-research-analysis[all]"

Unified-table coercion, validation, and derived-column helpers are already part
of the base install, so there is no separate ``table`` extra.

Use ``[maps,embeddings]`` for the default text-driven embedding-map API and CLI
path. ``[maps]`` alone is sufficient when you already have numeric vectors or
pass ``--feature-columns``. Dataset profiling, validation, and codebook helpers
require ``[data]``.

When working from a source checkout, replace ``design-research-analysis`` with
``.`` and add ``-e`` to install the same extras in editable mode.

Use :doc:`dependencies_and_extras` for family-level guidance and tradeoffs.
