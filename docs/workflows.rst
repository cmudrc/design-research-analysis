Workflows
=========

Unified Table Contract
----------------------

Required column:

- ``timestamp``

Strongly recommended columns:

- ``record_id``
- ``text``
- ``session_id``
- ``actor_id``
- ``event_type``

Optional common column:

- ``meta_json``

Loose schemas are supported. Missing fields can be derived with deterministic
mapper functions via :func:`design_research_analysis.derive_columns`.

Core Analysis Families
----------------------

Sequence
~~~~~~~~

- :func:`design_research_analysis.fit_markov_chain_from_table`
- :func:`design_research_analysis.fit_discrete_hmm_from_table`
- :func:`design_research_analysis.fit_text_gaussian_hmm_from_table`

Language
~~~~~~~~

- :func:`design_research_analysis.compute_language_convergence`
- :func:`design_research_analysis.compute_semantic_distance_trajectory`
- :func:`design_research_analysis.fit_topic_model`
- :func:`design_research_analysis.score_sentiment`

Embedding / Dimensionality Reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :func:`design_research_analysis.embed_records`
- :func:`design_research_analysis.reduce_dimensions`
- :func:`design_research_analysis.cluster_projection`

Statistics
~~~~~~~~~~

- :func:`design_research_analysis.compare_groups`
- :func:`design_research_analysis.fit_regression`
- :func:`design_research_analysis.fit_mixed_effects`

CLI
---

Validate a table:

.. code-block:: bash

   design-research-analysis validate-table \
     --input data/events.csv \
     --summary-json artifacts/validate.json

Run a sequence model:

.. code-block:: bash

   design-research-analysis run-sequence \
     --input data/events.csv \
     --summary-json artifacts/sequence.json \
     --mode markov

Run language analysis:

.. code-block:: bash

   design-research-analysis run-language \
     --input data/events.csv \
     --summary-json artifacts/language.json \
     --trajectory-csv artifacts/language_trajectory.csv
