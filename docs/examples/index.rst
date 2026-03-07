Examples Guide
==============

The examples in this repository are runnable research-oriented scripts. They are
designed to show not only API usage, but how the library fits into realistic
experimental workflows. Each example lists dependencies, expected scope, and
the primary concept it demonstrates.

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

Fit sequence models directly from event rows.

**Requires:** ``seq`` for full HMM coverage
**Runtime:** short
**Teaches:** token extraction from events, transition modeling, sequence summaries

Language Custom Embedder
~~~~~~~~~~~~~~~~~~~~~~~~

Run language-convergence workflows with custom embedding logic.

**Requires:** ``lang,embeddings``
**Runtime:** medium
**Teaches:** semantic trajectory analysis, embedder integration, text-study interpretation

Dimred PCA
~~~~~~~~~~

Project and cluster embeddings for structure inspection.

**Requires:** ``dimred``
**Runtime:** short
**Teaches:** projection choices, clustering interpretation, geometry-aware summaries

Stats Regression
~~~~~~~~~~~~~~~~

Run inferential modeling over event-derived variables.

**Requires:** ``stats,data``
**Runtime:** short
**Teaches:** model setup, coefficient interpretation, effect-focused reporting

Full Catalog
------------

.. toctree::
   :maxdepth: 1

   basic_usage
   dimred_pca
   lab_study_pipeline
   language_custom_embedder
   sequence_from_table
   stats_regression
   unified_table_validation
