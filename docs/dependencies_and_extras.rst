Dependencies and Extras
=======================

The project keeps runtime dependencies minimal and layers optional analysis
capabilities behind extras.

Core Install
------------

.. code-block:: bash

   pip install -e .

Runtime dependencies:

- ``numpy``
- ``matplotlib``

Development Install
-------------------

.. code-block:: bash

   make dev

This installs linting, typing, testing, docs, and release-check tooling.

Reproducible Install
--------------------

.. code-block:: bash

   make repro REPRO_EXTRAS="dev"

The frozen install uses ``uv.lock`` and pinned interpreter ``3.12.12``.

Extras Matrix
-------------

.. list-table::
   :header-rows: 1

   * - Extra
     - Purpose
     - Key packages
   * - ``table``
     - Semantic marker for table-first workflows
     - None
   * - ``data``
     - Dataframe profiling and schema helpers
     - ``pandas``
   * - ``seq``
     - Sequence modeling utilities
     - ``hmmlearn``, ``networkx``, ``scipy``
   * - ``embeddings``
     - Text embedding backends
     - ``sentence-transformers``
   * - ``lang``
     - Topic modeling and language helpers
     - ``scikit-learn``
   * - ``dimred``
     - Projection and manifold workflows
     - ``scikit-learn``, ``umap-learn``
   * - ``stats``
     - Statistical tests and model wrappers
     - ``scipy``, ``statsmodels``, ``pandas``
   * - ``all``
     - Convenience bundle of all optional analysis extras
     - All optional packages above

Recommended Profiles
--------------------

- Fast contributor loop: ``make dev``
- Sequence work: ``pip install -e ".[seq]"``
- Language + embeddings: ``pip install -e ".[lang,embeddings]"``
- Stats with dataframe helpers: ``pip install -e ".[stats,data]"``
- Full optional coverage: ``pip install -e ".[all]"``

Maintainer Release Baseline
---------------------------

Use this flow before tagging a release:

1. Use Python ``3.12.12`` (from ``.python-version``).
2. Regenerate lock data: ``make lock``.
3. Verify frozen install and checks: ``make repro REPRO_EXTRAS="dev"`` and ``make ci``.
4. Build release artifacts and validate metadata: ``make release-check``.
5. Commit lock/dependency updates before tagging.
