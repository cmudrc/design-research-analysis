Dependencies and Extras
=======================

Core Install
------------

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install design-research-analysis

Editable contributor setup:

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-analysis.git
   cd design-research-analysis
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"

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
     - DataFrame/file profiling, schema validation, and codebook workflows
   * - ``seq``
     - Sequence and HMM workflows
   * - ``embeddings``
     - Default sentence-transformer backend for text embedding
   * - ``lang``
     - Language/topic modeling workflows
   * - ``maps``
     - Non-PCA projection backends and clustering for embedding-map workflows
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

Base installs already support first-order Markov analysis, custom-embedder
language convergence, NumPy regression, PCA maps, and plotting. ``seq`` adds
HMM and graph backends. ``lang`` and ``embeddings`` add topic modeling and the
default sentence-transformer backend. ``maps`` adds manifold backends and
clustering. ``stats`` adds SciPy/statsmodels workflows. ``data`` is required by
``profile_dataframe``, ``validate_dataframe``, and ``generate_codebook``.

Recommended install profiles:

- HMM-focused studies: ``python -m pip install "design-research-analysis[seq]"``
- language + embedding studies: ``python -m pip install "design-research-analysis[lang,embeddings]"``
- text-driven embedding maps: ``python -m pip install "design-research-analysis[maps,embeddings]"``
- numeric-feature embedding maps: ``python -m pip install "design-research-analysis[maps]"``
- inference + dataset studies: ``python -m pip install "design-research-analysis[stats,data]"``
- broad analysis workstation setup: ``python -m pip install "design-research-analysis[all]"``

The ``run-embedding-maps`` CLI embeds ``text`` by default, so its default path
needs ``[maps,embeddings]``. With ``--feature-columns`` it skips text embedding,
and ``[maps]`` is enough. The dataset CLI commands ``profile-dataset``,
``validate-dataset``, and ``generate-codebook`` require ``[data]``.

If you are working from a local checkout instead of PyPI, replace
``design-research-analysis`` with ``.`` and add ``-e`` to install the same
extras in editable mode.

Release packaging validation is exposed via ``make release-check``.
