Docs Automation Baseline
========================

This page documents the shared docs and CI baseline for
``design-research-analysis``.

The analysis repo now matches the common module-repo posture: public docs stay
in sync with README and examples, example health is reported explicitly, and
release-facing surfaces are maintained with the same workflow split used across
the family.

Shared Module Baseline
----------------------

.. list-table::
   :header-rows: 1

   * - Concern
     - Local utility
     - Workflow owner
     - Baseline expectation
   * - Docs consistency
     - ``scripts/check_docs_consistency.py``
     - ``ci.yml``
     - README, docs landing pages, and generated example references stay aligned.
   * - Docstring policy
     - ``scripts/check_google_docstrings.py``
     - ``ci.yml``
     - Public APIs, scripts, and examples stay on the shared docstring standard.
   * - Coverage badge
     - ``scripts/generate_coverage_badge.py``
     - ``ci.yml``
     - Coverage status stays synchronized with the enforced 90% floor.
   * - Example docs generation
     - ``scripts/generate_example_docs.py``
     - ``examples.yml``
     - Checked-in examples remain represented in the published docs.
   * - Example reporting
     - ``scripts/generate_examples_metrics.py`` and ``scripts/generate_examples_badges.py``
     - ``examples.yml``
     - Example pass/fail and public-API coverage badges use the shared family format.
   * - Example boundary checks
     - ``scripts/check_example_api_coverage.py``
     - ``examples.yml``
     - Examples continue to exercise the documented public import surface.
   * - Release callout upkeep
     - ``scripts/update_release_readme.py``
     - ``update-release-readme.yml``
     - README release callouts stay aligned with the active monthly milestone.

Workflow Responsibilities
-------------------------

- ``ci.yml`` owns lint, type, test, coverage, docs-consistency, and docstring checks.
- ``examples.yml`` owns example execution, example-doc generation, and example-derived badge metrics.
- ``docs-pages.yml`` owns the published docs build.
- ``update-release-readme.yml`` owns README release-callout refresh.
- ``workflow.yml`` remains the aggregate maintainer workflow entry point.

Analysis-Specific Notes
-----------------------

``design-research-analysis`` does not need a repo-specific generator on the
scale of the problems catalog. Its repo-specific documentation work stays
centered on two maintained onboarding surfaces:

- :doc:`experiments_handoff` for exported ``events.csv`` inputs.
- :doc:`unified_table_schema` for the stable downstream column contract.

That means the shared baseline is intentionally enough here: the custom work is
in the content of the analysis docs, not in a one-off automation pipeline.

When To Update This Page
------------------------

Refresh this page whenever workflow ownership changes or when a new docs,
examples, or badge utility becomes part of the shared analysis maintainer loop.
