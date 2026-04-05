Dependencies and Extras
=======================

Core Install
------------

.. code-block:: bash

   python -m pip install --upgrade pip
   pip install design-research-analysis

Editable contributor setup:

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-analysis.git
   cd design-research-analysis
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e ".[dev]"

Or use:

.. code-block:: bash

   make dev

Maintainer workflows target Python ``3.12`` from ``.python-version``.

Extras Matrix
-------------

.. list-table::
   :header-rows: 1

   * - Extra
     - Purpose
   * - ``data``
     - DataFrame profiling and schema workflows
   * - ``seq``
     - Sequence and HMM workflows
   * - ``embeddings``
     - Sentence embedding backends
   * - ``lang``
     - Language/topic modeling workflows
   * - ``maps``
     - Embedding-map projection, manifold, and plotting workflows
   * - ``dimred``
     - Legacy alias for ``maps``
   * - ``stats``
     - Inferential and model-based statistics
   * - ``all``
     - Convenience bundle for all analysis extras
   * - ``dev``
     - Contributor tooling

Unified-table coercion, validation, and derived-column helpers are part of the
base install, so there is no separate ``table`` extra to add.

``seq`` is usually the first add-on for event-transition studies. ``lang`` and
``embeddings`` are most useful for discourse and semantic analyses. ``maps`` is
best when structural embedding comparisons or trajectory overlays are central.
``stats`` is best when inferential modeling is central. ``all`` is appropriate
when building a full local research environment.

Recommended install profiles:

- sequence-focused studies: ``pip install "design-research-analysis[seq]"``
- language + embedding studies: ``pip install "design-research-analysis[lang,embeddings]"``
- embedding-map studies: ``pip install "design-research-analysis[maps]"``
- inference-heavy studies: ``pip install "design-research-analysis[stats,data]"``
- broad analysis workstation setup: ``pip install "design-research-analysis[all]"``

If you are working from a local checkout instead of PyPI, replace
``design-research-analysis`` with ``.`` to install the same extras in editable mode.

Release packaging validation is exposed via ``make release-check``.
