Dependencies and Extras
=======================

Core Install
------------

.. code-block:: bash

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
   * - ``dimred``
     - Projection and manifold workflows
   * - ``stats``
     - Inferential and model-based statistics
   * - ``all``
     - Convenience bundle for all analysis extras
   * - ``dev``
     - Contributor tooling

``seq`` is usually the first add-on for event-transition studies. ``lang`` and
``embeddings`` are most useful for discourse and semantic analyses. ``stats`` is
best when inferential modeling is central. ``all`` is appropriate when building
a full local research environment.

Recommended install profiles:

- sequence-focused studies: ``pip install -e ".[seq]"``
- language + embedding studies: ``pip install -e ".[lang,embeddings]"``
- inference-heavy studies: ``pip install -e ".[stats,data]"``
- broad analysis workstation setup: ``pip install -e ".[all]"``

Reproducible and release flows are exposed via ``make repro``, ``make lock``,
and ``make release-check``.
