Examples
========

The examples in this repository are runnable research-oriented scripts. They are
designed to show not only API usage, but how the library fits into realistic
experimental workflows. The featured examples below list dependencies,
expected scope, and the primary concept they demonstrate.

Featured Examples
-----------------

Unified Table Validation
~~~~~~~~~~~~~~~~~~~~~~~~

Validate and normalize a unified event table before modeling.

**Requires:** base install
**Runtime:** short
**Teaches:** schema checks, missing-column handling, derivation-first analysis setup

Sequence From Table
~~~~~~~~~~~~~~~~~~~

Fit a first-order Markov chain directly from event rows.

**Requires:** base install
**Runtime:** short
**Teaches:** token extraction from events, transition modeling, sequence summaries

Language Custom Embedder
~~~~~~~~~~~~~~~~~~~~~~~~

Run language-convergence workflows with custom embedding logic.

**Requires:** base install; the script supplies a custom embedder
**Runtime:** medium
**Teaches:** semantic trajectory analysis, embedder integration, text-study interpretation

Embedding Maps Trajectories
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot trace-aware embedding maps with scalar value overlays.

**Requires:** base install; ``maps`` adds optional comparison methods
**Runtime:** short
**Teaches:** multi-method map comparison, trace overlays, and scalar-colored trajectories

Idea Space Metrics
~~~~~~~~~~~~~~~~~~

Compute idea-space coverage and trajectory summaries, then render the main plots.

**Requires:** base install
**Runtime:** short
**Teaches:** coverage metrics, divergence diagnostics, and top-level visualization helpers

Stats Regression
~~~~~~~~~~~~~~~~

Fit an ordinary least-squares model over event-derived variables.

**Requires:** base install
**Runtime:** short
**Teaches:** model setup, coefficient interpretation, effect-focused reporting

Full Catalog
------------

.. toctree::
   :maxdepth: 1

   basic_usage
   condition_pair_significance
   embedding_maps_trajectories
   experiment_artifacts_handoff
   idea_space_metrics
   lab_study_pipeline
   language_custom_embedder
   mechanical_design_review_analysis
   paper_contributions
   sequence_from_table
   stats_interrater_reliability
   stats_regression
   unified_table_validation
   verified_paper_bundle
