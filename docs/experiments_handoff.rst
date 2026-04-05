Experiments-To-Analysis Handoff
===============================

Use this guide when you already have study artifacts exported from
``design-research-experiments`` and want the shortest reliable path into
analysis.

The analysis-owned cross-library handoff lives in
``design_research_analysis.integration``.

Why This Handoff Exists
-----------------------

``design-research-experiments`` exports canonical study artifacts, including
``events.csv``, ``runs.csv``, and ``evaluations.csv``. In
``design-research-analysis``, ``events.csv`` is the primary first-class input
for validation and downstream workflows.

For background on the study-side workflow, see the
`design-research-experiments typical workflow <https://cmudrc.github.io/design-research-experiments/typical_workflow.html>`_
and
`reference overview <https://cmudrc.github.io/design-research-experiments/reference/index.html>`_.
For the canonical export contract itself, see the
`design-research-experiments artifact contract <https://cmudrc.github.io/design-research-experiments/artifact_contract.html>`_.

Canonical Input Files
---------------------

- ``events.csv``: event-level analysis input for validation, sequence, language,
  and embedding-map workflows.
- ``runs.csv``: run-level study context such as condition, model, seed, status,
  and outcome metadata.
- ``evaluations.csv``: evaluator outputs keyed to a run, such as scores or
  rubric metrics.

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

Validation And Derivation In Python
-----------------------------------

.. code-block:: python

   from design_research_analysis import fit_markov_chain_from_table
   from design_research_analysis.integration import (
       load_experiment_artifacts,
       validate_experiment_events,
   )

   artifacts = load_experiment_artifacts("study-output")
   report = validate_experiment_events("study-output/events.csv")
   if not report.is_valid:
       raise RuntimeError("; ".join(report.errors))

   result = fit_markov_chain_from_table(artifacts["events.csv"])
   print(result.states)

Column Expectations In The Export Handoff
-----------------------------------------

Required for unified-table validation:

- ``timestamp``

Strongly recommended in exported ``events.csv``:

- ``record_id``
- ``text``
- ``session_id``
- ``actor_id``
- ``event_type``

Optional but commonly present:

- ``meta_json``
- ``run_id``

Derived-column guidance:

- Derive ``actor_id`` when the experiment trace uses another participant field.
- Derive ``event_type`` when raw observations need normalization into a shared
  event vocabulary.
- Derive ``record_id`` when upstream events are otherwise stable but unlabeled.

Rejoin Study Context
--------------------

After analysis, rejoin event-level outputs back to study context.

Preferred join path:

- If ``events.csv`` includes ``run_id``, join on
  ``events.run_id -> runs.run_id`` and
  ``events.run_id -> evaluations.run_id``.

Fallback join path:

- If ``run_id`` is absent, join on
  ``events.session_id -> runs.run_id`` and
  ``events.session_id -> evaluations.run_id``.
  The experiments exporter defaults ``session_id`` to the run id when a session
  id is otherwise missing.

Compare Condition Metrics From Exports
--------------------------------------

Use the higher-level stats helpers when you want pairwise condition
comparisons over canonical experiment exports without custom joins.

.. code-block:: python

   from design_research_analysis import (
       build_condition_metric_table,
       compare_condition_pairs,
   )

   condition_metric_rows = build_condition_metric_table(
       runs_rows,
       metric="market_share_proxy",
       condition_column="selection_strategy",
       conditions=conditions_rows,
       evaluations=evaluations_rows,
   )

   report = compare_condition_pairs(
       condition_metric_rows,
       condition_pairs=[
           ("profit_focus_prompt", "neutral_prompt"),
           ("neutral_prompt", "random_selection"),
       ],
       alternative="greater",
       seed=17,
   )

   print(report.render_brief())
   print(report.to_significance_rows())

Related Docs
------------

- :doc:`quickstart`
- :doc:`typical_workflow`
- :doc:`unified_table_schema`
- :doc:`cli_reference`
