design-research-analysis
========================

Reusable analysis workflows for design-research event data.

What This Library Does
----------------------

``design-research-analysis`` supports sequence analysis, language analysis,
dimensionality reduction, and statistical modeling over unified event-table
inputs. It is intended for recurring research workflows where validation,
provenance, and repeatability are first-order concerns.

Highlights
----------

- Unified table schema
- Sequence modeling
- Language analysis
- Embeddings and dimred
- Statistical workflows
- Provenance capture

The library assumes real datasets are messy. Unified-table validation and
column derivation are therefore core features, not pre-processing footnotes.
They make downstream analysis functions composable and reproducible.

Typical Workflow
----------------

1. Load event-table records.
2. Validate and derive required analytical columns.
3. Select one or more analysis families.
4. Run models and write summaries/artifacts.
5. Interpret outputs in study context and preserve provenance.

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
   :caption: Documentation
   :hidden:

   quickstart
   installation
   concepts
   typical_workflow
   examples/index
   api

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   dependencies_and_extras

.. toctree::
   :maxdepth: 2
   :caption: Additional Guides
   :hidden:

   unified_table_schema
   workflows
   analysis_recipes
   cli_reference
