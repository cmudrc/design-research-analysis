Docs Automation Baseline
========================

This page documents the shared docs and CI baseline for
``design-research-analysis``.

The analysis repo now matches the common module-repo posture: public docs stay
in sync with README and examples, and example health is reported explicitly.
Release state is tracked through package versions, GitHub Releases, and focused
PRs rather than generated README callouts or default monthly milestones.

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
     - Internal navigation resolves, every top-level ``__all__`` export appears
       in the API inventory, and every checked-in example appears in its README.
   * - Docstring policy
     - ``scripts/check_google_docstrings.py``
     - ``ci.yml``
     - Public APIs, scripts, and examples stay on the shared docstring standard.
   * - Coverage badge
     - ``scripts/generate_coverage_badge.py``
     - ``ci.yml``
     - Coverage status stays synchronized with the enforced 95% floor.
   * - Example docs generation
     - ``scripts/generate_example_docs.py``
     - ``ci.yml`` and ``docs-pages.yml``
     - Checked-in examples remain represented in the published docs.
   * - Example reporting
     - ``scripts/generate_examples_metrics.py`` and ``scripts/generate_examples_badges.py``
     - ``ci.yml``
     - Example pass/fail and public-API coverage badges use the shared family format.
   * - Example boundary checks
     - ``scripts/check_example_api_coverage.py``
     - ``examples.yml`` and ``ci.yml``
     - The runnable suite continues to reference the supported top-level import
       surface at the configured coverage threshold.

Workflow Responsibilities
-------------------------

- ``ci.yml`` owns lint, type, test, coverage, generated-doc consistency,
  docstring checks, example boundary checks, and example-derived badge metrics.
- ``examples.yml`` owns the standalone example-execution and public-API coverage checks.
- ``docs-pages.yml`` owns example-doc generation and the strict published docs build.
  Link checking remains the explicit ``make docs-linkcheck`` target.
- ``workflow.yml`` owns release builds and authorized PyPI publishing; it is not
  an aggregate validation workflow.

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
