API
===

This page documents the supported top-level public API from
``design_research_analysis.__all__``.

Top-level groups:

- Module facades: ``dataset``, ``dimred``, ``language``, ``runtime``, ``sequence``, ``stats``
- Unified table contracts: ``UnifiedTableConfig``, ``UnifiedTableValidationReport``,
  ``coerce_unified_table``, ``derive_columns``, ``validate_unified_table``
- Sequence: ``MarkovChainResult``, ``DiscreteHMMResult``, ``GaussianHMMResult``,
  ``DecodeResult``, ``fit_markov_chain_from_table``, ``fit_discrete_hmm_from_table``,
  ``fit_text_gaussian_hmm_from_table``, ``decode_hmm``, ``plot_transition_matrix``,
  ``plot_state_graph``
- Language: ``compute_language_convergence``, ``compute_semantic_distance_trajectory``,
  ``fit_topic_model``, ``score_sentiment``
- Dimensionality reduction: ``embed_records``, ``reduce_dimensions``,
  ``cluster_projection``
- Statistics: ``compare_groups``, ``fit_regression``, ``fit_mixed_effects``,
  ``permutation_test``, ``bootstrap_ci``, ``rank_tests_one_stop``,
  ``estimate_sample_size``, ``power_curve``, ``minimum_detectable_effect``
- Dataset + runtime: ``profile_dataframe``, ``validate_dataframe``, ``generate_codebook``,
  ``capture_run_context``, ``attach_provenance``, ``is_notebook``,
  ``is_google_colab``, ``write_run_manifest``

.. automodule:: design_research_analysis
   :members:
   :undoc-members:
   :show-inheritance:
