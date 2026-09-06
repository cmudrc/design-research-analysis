Analysis Results and Paper Contributions
========================================

``design-research-analysis`` can preserve an executed analysis as a portable,
versioned result and translate it into deterministic Methods and Results
support. This layer does not replace the package's existing result objects and
does not generate a manuscript. It records what ran, which evidence it used,
and which reporting facts remain unresolved.

The result and contribution contracts are both version ``0.1.0``. They are
additive to the canonical experiment artifact schema.

Build a Result Record
---------------------

Run the normal analysis helper first, then wrap its result:

.. code-block:: python

   import design_research_analysis as dran

   fitted = dran.fit_regression(
       [[0.0], [1.0], [2.0]],
       [1.0, 1.5, 2.0],
       feature_names=["iteration"],
   )
   record = dran.build_analysis_result(
       fitted,
       analysis_id="h1-regression",
       analysis_plan_ids=("plan-h1",),
       hypothesis_ids=("H1",),
       candidate_run_ids=("run-1", "run-2", "run-3"),
       included_run_ids=("run-1", "run-2"),
       exclusions=(
           dran.AnalysisExclusion(
               run_id="run-3",
               reason="The prespecified evaluator output was missing.",
           ),
       ),
       assumptions=(
           dran.AnalysisCheck(
               check_id="linearity",
               status="passed",
               detail="The residual plot was inspected.",
           ),
       ),
       evidence_refs=(
           "artifacts/runs/run-1/run.json",
           "artifacts/runs/run-2/run.json",
       ),
       source_api="fit_regression",
   )

``build_analysis_result`` recognizes the package's regression, group and
condition comparison, reliability, Markov-chain, embedding-map, language
convergence, mixed-effects, and dataset-profile result shapes. A custom mapping
remains supported when its caller supplies an explicit namespaced
``method_id``.

The record distinguishes candidate, included, excluded, and unaccounted runs.
Every exclusion needs a reason. A complete result needs at least one durable
``evidence_ref``; in-memory configuration alone is not analyzed evidence.

Persist and Reload
------------------

Persistence is explicit:

.. code-block:: python

   result_path = dran.write_analysis_result(record, output_dir="study-output")
   reloaded = dran.load_analysis_result(result_path)

The writer atomically creates:

.. code-block:: text

   study-output/
     artifacts/
       analysis/
         results/
           h1-regression.json

Existing records are not overwritten unless ``overwrite=True`` is supplied.
Analysis identifiers and table/figure paths are validated so absolute paths and
parent-directory escapes cannot enter a later portable bundle.

Collect Paper Contributions
---------------------------

.. code-block:: python

   packet = dran.collect_analysis_paper_contributions(reloaded)

The returned JSON-compatible packet matches the ``0.1.0`` component contract
accepted by ``design-research-experiments``. It contains:

- curated Methods wording owned by the analysis implementation;
- restrained Results statements derived from recorded statistics;
- hypothesis, analysis-plan, run-inclusion, and exclusion links;
- table and figure contributions using safe relative paths;
- curated references only where the method has a concrete source; and
- reporting gaps for failed analyses, failed checks, missing uncertainty,
  unattached figures, or unaccounted candidate runs.

A p-value is never converted into a decision unless the record includes an
explicit ``decision_threshold``. Regression point estimates are not described
as inferential evidence when standard errors or p-values are unavailable.
Planned or failed analyses produce TODOs rather than Results claims.

Supported Initial Renderers
---------------------------

- ordinary least-squares regression;
- group and condition comparisons;
- fitted and compared Markov chains;
- Cohen, Fleiss, and Krippendorff inter-rater reliability;
- mixed-effects fit summaries;
- dataset profiles;
- embedding maps and their attached figures; and
- language-convergence summaries.

Unknown method identifiers remain valid for forward compatibility. They
receive a generic Methods block and a visible TODO requesting a curated Results
renderer instead of guessed prose.
