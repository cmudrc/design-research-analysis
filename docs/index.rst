design-research-analysis
========================

The analysis layer for reproducible design-research event data.

``design-research-analysis`` supports sequence analysis, language analysis,
dimensionality reduction, and statistical modeling over unified event-table
inputs. It is built for recurring research workflows where validation,
provenance, and repeatability are first-order concerns.

Unified-table validation and column derivation are core features, not
pre-processing footnotes. They make downstream analyses composable,
reproducible, and easier to compare across studies.

.. raw:: html

   <div class="drc-badge-row">
     <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-analysis/actions/workflows/ci.yml">
       <img alt="CI" src="https://github.com/cmudrc/design-research-analysis/actions/workflows/ci.yml/badge.svg">
     </a>
     <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-analysis/actions/workflows/ci.yml">
       <img alt="Coverage" src="https://raw.githubusercontent.com/cmudrc/design-research-analysis/main/.github/badges/coverage.svg">
     </a>
     <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-analysis/actions/workflows/examples.yml">
       <img alt="Examples Passing" src="https://raw.githubusercontent.com/cmudrc/design-research-analysis/main/.github/badges/examples-passing.svg">
     </a>
     <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-analysis/actions/workflows/examples.yml">
       <img alt="Public API In Examples" src="https://raw.githubusercontent.com/cmudrc/design-research-analysis/main/.github/badges/examples-api-coverage.svg">
     </a>
     <a class="drc-badge-link" href="https://github.com/cmudrc/design-research-analysis/actions/workflows/docs-pages.yml">
       <img alt="Docs" src="https://github.com/cmudrc/design-research-analysis/actions/workflows/docs-pages.yml/badge.svg">
     </a>
   </div>

.. note::

   **Start with** :doc:`quickstart` to validate a first event table, run a
   representative analysis pass, and get the package into a reproducible local
   loop before diving into the broader workflow and API material.

Guides
------

Learn the table contract, setup flow, and analysis workflow patterns that
shape a stable research pipeline.

- :doc:`quickstart`
- :doc:`installation`
- :doc:`concepts`
- :doc:`typical_workflow`
- :doc:`workflows`
- :doc:`analysis_recipes`

Examples
--------

Browse runnable examples that show the public API in action across the major
analysis families.

- :doc:`examples/index`
- :doc:`examples/basic_usage`
- :doc:`examples/unified_table_validation`
- :doc:`examples/sequence_from_table`
- :doc:`examples/dimred_pca`
- :doc:`examples/stats_regression`

Reference
---------

Look up the stable import surface, CLI behavior, and documentation for the
core table contract and optional extras.

- :doc:`api`
- :doc:`cli_reference`
- :doc:`unified_table_schema`
- :doc:`dependencies_and_extras`

Integration With The Ecosystem
------------------------------

The Design Research Collective maintains a modular ecosystem of libraries for
studying human and AI design behavior.

- **design-research-agents** implements AI participants, workflows, and tool-using reasoning patterns.
- **design-research-problems** provides benchmark design tasks, prompts, grammars, and evaluators.
- **design-research-analysis** analyzes the traces, event tables, and outcomes generated during studies.
- **design-research-experiments** sits above the stack as the study-design and orchestration layer, defining hypotheses, factors, conditions, replications, and artifact flows across agents, problems, and analysis.

Together these libraries support end-to-end design research pipelines, from
study design through execution and interpretation.

.. image:: _static/ecosystem-platform.svg
   :alt: Ecosystem diagram showing experiments above agents, problems, and analysis.
   :width: 100%
   :align: center

Start Here
----------

- :doc:`quickstart`
- :doc:`installation`
- :doc:`concepts`
- :doc:`typical_workflow`
- :doc:`examples/index`
- :doc:`api`
- `CONTRIBUTING.md <https://github.com/cmudrc/design-research-analysis/blob/main/CONTRIBUTING.md>`_

.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart
   installation
   concepts
   typical_workflow
   workflows
   analysis_recipes
   examples/index
   api
   unified_table_schema
   cli_reference
   dependencies_and_extras
