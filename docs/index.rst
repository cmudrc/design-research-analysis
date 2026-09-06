design-research-analysis
========================

The analysis layer for reproducible design-research event data.

What This Library Does
----------------------

``design-research-analysis`` is the analysis and interpretation layer in the
CMU Design Research Collective design-research ecosystem. It supports sequence
analysis, language analysis, embedding maps, and statistical modeling over
unified event tables and canonical experiment artifacts. It is built for
recurring research workflows where validation, provenance, and repeatability
are first-order concerns.

Unified-table validation and column derivation are core features, not
pre-processing footnotes. They make downstream analyses composable,
reproducible, and easier to compare across studies.

.. container:: drc-home-badges

   .. raw:: html

      <div class="drc-badge-row">
        <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-analysis/actions/workflows/ci.yml">
          <img alt="CI" src="https://github.com/cmudrc/design-research-analysis/actions/workflows/ci.yml/badge.svg">
        </a>
        <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-analysis/actions/workflows/ci.yml">
          <img alt="Coverage" src="https://raw.githubusercontent.com/cmudrc/design-research-analysis/HEAD/.github/badges/coverage.svg">
        </a>
        <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-analysis/actions/workflows/examples.yml">
          <img alt="Examples Passing" src="https://raw.githubusercontent.com/cmudrc/design-research-analysis/HEAD/.github/badges/examples-passing.svg">
        </a>
        <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-analysis/actions/workflows/examples.yml">
          <img alt="API in Examples" src="https://raw.githubusercontent.com/cmudrc/design-research-analysis/HEAD/.github/badges/examples-api-coverage.svg">
        </a>
        <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-analysis/actions/workflows/docs-pages.yml">
          <img alt="Docs" src="https://github.com/cmudrc/design-research-analysis/actions/workflows/docs-pages.yml/badge.svg">
        </a>
        <a class="drc-badge-link" href="https://pypi.org/project/design-research-analysis/">
          <img alt="PyPI Version" src="https://img.shields.io/pypi/v/design-research-analysis.svg">
        </a>
        <a class="drc-badge-link" href="https://pypi.org/project/design-research-analysis/">
          <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/design-research-analysis.svg">
        </a>
      </div>

Quality Signals
---------------

- ``Coverage`` reports total line coverage for the default deterministic test
  suite; CI requires at least 95%.
- ``Examples Passing`` reports checked-in example scripts that execute
  successfully in the examples workflow.
- ``API in Examples`` reports curated top-level ``__all__`` exports referenced
  by runnable examples. ``N/N`` means every supported top-level export appears
  in at least one example, and CI requires 100%.

Run ``make coverage``, ``make examples-test``, and ``make examples-coverage``
to reproduce these checks locally.

Highlights
----------

- Unified-table coercion, validation, and mapper-driven derived columns
- Dataset profiling, schema checks, and codebook generation
- Sequence analysis for Markov chains and Hidden Markov Models
- Language analysis for semantic convergence, topic discovery, and sentiment
- Embedding maps and clustering for embedding-space inspection
- Statistical workflows for comparisons, regression, mixed effects, and power
- Runtime provenance capture for reproducible study artifacts
- Top-level artifact handoff helpers for experiment exports
- Portable analysis results and evidence-linked paper contributions

Typical Workflow
----------------

1. Start from a unified event table or top-level artifact helpers over an
   exported ``design-research-experiments`` study-output directory.
2. Validate and, when needed, derive missing analysis columns.
3. Run sequence, language, embedding-map, and/or statistical workflows.
4. Persist JSON summaries, CSV exports, and provenance manifests.
5. Rejoin findings to ``runs.csv`` and ``evaluations.csv`` for study context.

.. container:: drc-home-callout

   .. note::

      **Start with** :doc:`quickstart` for the shortest runnable path, or
      :doc:`experiments_handoff` if you already have ``events.csv`` from
      ``design-research-experiments``.

Guides
------

Learn the table model, setup flow, and repeatable analysis patterns that shape
a stable downstream research pipeline.

- :doc:`guides`
- :doc:`installation`
- :doc:`quickstart`
- :doc:`concepts`
- :doc:`experiments_handoff`
- :doc:`paper_contributions`
- :doc:`typical_workflow`
- :doc:`workflows`
- :doc:`analysis_recipes`
- :doc:`unified_table_schema`

Examples
--------

Browse runnable examples covering the major analysis surfaces.

- :doc:`examples/index`

Reference
---------

Look up the stable import surface, CLI behavior, and dependency guidance for
repeatable analysis environments.

- :doc:`reference/index`
- :doc:`api`
- :doc:`cli_reference`
- :doc:`dependencies_and_extras`

Architecture: Two Complementary Views
-------------------------------------

**Control topology:** Problems and Agents are peer study inputs. Experiments
owns study design and coordinates their execution, then defines the artifact
handoff to Analysis.

**Runtime and data flow:** Problems + Agents → Experiments artifact set →
Analysis → evidence that can refine the next study protocol.

These are two views of the same package family, not an installation order. The
umbrella routes imports and pins a tested combination; implementation stays
with the package that owns each behavior. See the umbrella
`compatibility and package status <https://cmudrc.github.io/design-research/compatibility.html>`_
for the tested family combination.

.. container:: drc-home-ecosystem

   .. image:: _static/ecosystem-platform.svg
      :alt: Two-view diagram showing the control topology and runtime data flow across Problems, Agents, Experiments, and Analysis.
      :class: dark-light drc-ecosystem-figure
      :width: 100%
      :align: center

Ecosystem Packages
------------------

- **Problems** — tasks, prompts, grammars, benchmarks, and evaluators:
  `documentation <https://cmudrc.github.io/design-research-problems/>`__
- **Agents** — AI participants, workflows, tools, and traceable reasoning:
  `documentation <https://cmudrc.github.io/design-research-agents/>`__
- **Experiments** — hypotheses, factors, conditions, replications, execution,
  and artifact export:
  `documentation <https://cmudrc.github.io/design-research-experiments/>`__
- **Analysis** — validation, transformation, statistics, and visualization of
  study artifacts: :doc:`guides`
- **Umbrella** — routed imports, learning paths, and tested compatibility:
  `documentation <https://cmudrc.github.io/design-research/>`__

Start Here
----------

- :doc:`guides`
- :doc:`installation`
- :doc:`quickstart`
- :doc:`concepts`
- :doc:`experiments_handoff`
- :doc:`paper_contributions`
- :doc:`typical_workflow`
- :doc:`examples/index`
- :doc:`api`
- :doc:`vscode_start`
- :doc:`automation_baseline`
- `CONTRIBUTING.md <https://github.com/cmudrc/design-research-analysis/blob/HEAD/CONTRIBUTING.md>`_

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   guides
   paper_contributions

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   reference/index

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   Contributing <https://github.com/cmudrc/design-research-analysis/blob/HEAD/CONTRIBUTING.md>
