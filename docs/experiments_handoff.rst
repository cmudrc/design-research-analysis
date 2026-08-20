Experiments-To-Analysis Handoff
===============================

Use this guide when you already have study artifacts exported from
``design-research-experiments`` and want the shortest reliable path into
analysis.

The analysis-owned cross-library handoff is exposed through top-level
artifact helpers in ``design_research_analysis``.

Why This Handoff Exists
-----------------------

``design-research-experiments`` exports a canonical study-artifact directory.
Its analysis-facing files are ``manifest.json``, ``conditions.csv``,
``runs.csv``, ``events.csv``, and ``evaluations.csv``. In
``design-research-analysis``, ``events.csv`` is the primary first-class input
for validation and downstream workflows.

For background on the study-side workflow, see the
`design-research-experiments typical workflow <https://cmudrc.github.io/design-research-experiments/typical_workflow.html>`_
and
`reference overview <https://cmudrc.github.io/design-research-experiments/reference/index.html>`_.
For the canonical export contract itself, see the
`design-research-experiments artifact contract <https://cmudrc.github.io/design-research-experiments/artifact_contract.html>`_.

The stable handoff unit is the exported study-output directory. For standard
workflows, prefer artifact-first helpers that accept that directory and perform
the run, condition, event, and evaluation joins internally.

Artifact Contract Version
-------------------------

The current producer schema is ``0.2.0``. The analysis adapter accepts both
``0.2.0`` and the earlier ``0.1.0`` schema for backward compatibility. Every
artifact-first helper validates ``manifest.json`` before
reading tables and raises a clear error for missing or unsupported versions.
This keeps schema changes explicit without importing
``design-research-experiments`` or coupling analysis code to its internal
Python models.

Canonical Input Files
---------------------

- ``manifest.json``: artifact-set version authority and study identity.
- ``conditions.csv``: one row per condition with fixed ``study_id``,
  ``condition_id``, ``admissible``, ``constraint_messages``, and
  ``assignment_meta_json`` columns, followed by one flat column per factor and
  ``block_<name>`` columns for block assignments.
- ``events.csv``: event-level analysis input for validation, sequence, language,
  and embedding-map workflows.
- ``runs.csv``: run-level study context such as condition, model, seed, status,
  and outcome metadata.
- ``evaluations.csv``: evaluator outputs keyed to a run, such as scores or
  rubric metrics.

``load_experiment_artifacts`` loads all five analysis-facing files. Individual
artifact helpers load the manifest plus the subset needed for their join.

Start With ``events.csv``
-------------------------

Validate the exported table before running analysis-family commands.

.. code-block:: bash

   design-research-analysis validate-table \
     --input study-output/events.csv \
     --summary-json artifacts/validate_table.json

Then run one downstream analysis workflow on the same artifact input.

.. code-block:: bash

   design-research-analysis run-sequence \
     --input study-output/events.csv \
     --summary-json artifacts/sequence.json \
     --mode markov

You can use the same validated ``events.csv`` input for language, embedding-map, and
stats commands depending on the study question.

For standard Python workflows, start with the package-level artifact helpers:

.. code-block:: python

   import design_research_analysis as dran

   report = dran.validate_experiment_events("study-output/events.csv")
   metric_rows = dran.build_condition_metric_table_from_artifacts(
       "study-output",
       metric="quality_score",
       condition_column="agent_treatment",
   )
   print(report.is_valid, len(metric_rows))

Validation And Derivation In Python
-----------------------------------

.. code-block:: python

   import design_research_analysis as dran

   report = dran.validate_experiment_events("study-output/events.csv")
   if not report.is_valid:
       raise RuntimeError("; ".join(report.errors))

   chains = dran.fit_markov_chains_from_artifacts(
       "study-output",
       condition_column="agent_treatment",
   )
   print(chains["planner"].states)

Loose Validation Versus Artifact Joins
--------------------------------------

These expectations are the downstream-facing slice of the
`artifact contract <https://cmudrc.github.io/design-research-experiments/artifact_contract.html>`_.

``validate_experiment_events`` deliberately applies the package's generic
unified-table validator. Its only required column is:

- ``timestamp``

The following columns are strongly recommended by that generic validator and
are required by particular downstream analysis families:

- ``record_id``
- ``text``
- ``session_id``
- ``actor_id``
- ``event_type``

The experiment schema ``0.2.0`` export contract also includes:

- ``meta_json``

``run_id`` is not a required ``events.csv`` header in producer schema
``0.2.0``. Do not interpret a valid ``validate_experiment_events`` report as
proof that an artifact-directory join can succeed; that helper checks event
table shape, not referential integrity against ``runs.csv``.

Derived-column guidance:

- Derive ``actor_id`` when the experiment trace uses another participant field.
- Derive ``event_type`` when raw observations need normalization into a shared
  event vocabulary.
- Derive ``record_id`` when upstream events are otherwise stable but unlabeled.

Treat the loose table and strict handoff differently in maintainer docs and
downstream code:

- The generic required column gates whether a loose event table is valid enough
  to proceed at all.
- Strongly recommended columns keep the exported events useful across sequence,
  language, embedding, and statistics workflows without custom preprocessing.
- Derived columns are the sanctioned fallback when upstream traces are stable
  but still need normalization into the shared analysis vocabulary.

Event-to-Run Join Precondition
------------------------------

Artifact-first event helpers such as ``build_event_table_from_artifacts`` must
resolve every event to ``runs.csv``. For each event they use a non-empty
``run_id`` when present; otherwise they fall back to ``session_id`` and require
that value to equal a ``runs.csv.run_id``. Unknown or blank identifiers raise a
``ValueError``.

The built-in experiments agent adapter supplies ``run_id`` and normally uses
the run identifier as ``session_id``. The experiments exporter does not
guarantee ``run_id`` for arbitrary observations supplied directly by callers,
and callers may provide unrelated session identifiers. Such producers must set
``run_id`` themselves or keep ``session_id == runs.csv.run_id``. This is a
current code-level contract gap, not a promise the documentation can repair.

Condition Context Columns
-------------------------

``build_event_table_from_artifacts`` joins the matching ``conditions.csv`` row
onto each event. By default this exposes the flat factor columns and
``block_<name>`` assignment columns as analysis context; fixed bookkeeping
columns such as ``assignment_meta_json`` are excluded from the default context
set. Pass ``condition_columns=...`` when a workflow needs an explicit subset.

Artifact-First Workflows
------------------------

The artifact helpers hide canonical-table joins for common study questions.
Use these first, then drop down to table-level APIs only when you need custom
feature engineering.

Compare Condition Metrics From Exports
--------------------------------------

Use the higher-level stats helpers when you want pairwise condition
comparisons over canonical experiment exports without custom joins.

.. code-block:: python

   from design_research_analysis import (
       compare_condition_pairs_from_artifacts,
   )

   report = compare_condition_pairs_from_artifacts(
       "study-output",
       metric="market_share_proxy",
       condition_column="selection_strategy",
       condition_pairs=[
           ("profit_focus_prompt", "neutral_prompt"),
           ("neutral_prompt", "random_selection"),
       ],
       alternative="greater",
       seed=17,
   )

   print(report.render_brief())
   print(report.to_significance_rows())

Compare Markov Chains Across Agent Treatments
---------------------------------------------

.. code-block:: python

   from design_research_analysis import compare_markov_chains_from_artifacts

   comparison = compare_markov_chains_from_artifacts(
       "study-output",
       condition_column="agent_treatment",
       left_condition="planner_agent",
       right_condition="baseline_agent",
   )

   print(comparison.to_dict())

Fit Regressions Across Sweeps And Factorial Designs
---------------------------------------------------

.. code-block:: python

   from design_research_analysis import fit_regression_from_artifacts

   result = fit_regression_from_artifacts(
       "study-output",
       outcome="rubric_score",
       predictors=("model_size_b", "task_family", "agent_treatment"),
       categorical_predictors=("task_family", "agent_treatment"),
   )

   print(result.coefficients)

Related Docs
------------

- :doc:`quickstart`
- :doc:`typical_workflow`
- :doc:`unified_table_schema`
- :doc:`cli_reference`
