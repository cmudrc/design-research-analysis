API
===

This page documents the supported top-level public API from
``design_research_analysis.__all__``.

Top-level groups:

- Package metadata: ``__version__``
- Comparison: ``ComparisonResult``
- Module facades: ``dataset``, ``embedding_maps``, ``integration``, ``language``,
  ``runtime``, ``sequence``, ``stats``, ``visualization``
- Unified table contracts: ``UnifiedTableConfig``, ``UnifiedTableValidationReport``,
  ``coerce_unified_table``, ``derive_columns``, ``validate_unified_table``
- Sequence: ``MarkovChainResult``, ``DiscreteHMMResult``, ``GaussianHMMResult``,
  ``DecodeResult``, ``fit_markov_chain_from_table``, ``fit_discrete_hmm_from_table``,
  ``fit_text_gaussian_hmm_from_table``, ``decode_hmm``, ``plot_transition_matrix``,
  ``plot_state_graph``
- Visualization: ``plot_design_process_timeline``, ``plot_idea_trajectory``,
  ``plot_convergence_curve``
- Language: ``compute_language_convergence``, ``compute_semantic_distance_trajectory``,
  ``fit_topic_model``, ``score_sentiment``
- Embedding maps: ``EmbeddingResult``, ``EmbeddingMapResult``,
  ``embed_records``, ``build_embedding_map``, ``cluster_embedding_map``,
  ``compare_embedding_maps``, ``compute_design_space_coverage``,
  ``compute_idea_space_trajectory``, ``compute_divergence_convergence``,
  ``plot_embedding_map``, ``plot_embedding_map_grid``
- Statistics: ``compare_groups``, ``fit_regression``, ``fit_mixed_effects``,
  ``permutation_test``, ``build_condition_metric_table``,
  ``compare_condition_pairs``, ``bootstrap_ci``, ``rank_tests_one_stop``,
  ``estimate_sample_size``, ``power_curve``, ``minimum_detectable_effect``
- Dataset + runtime: ``profile_dataframe``, ``validate_dataframe``, ``generate_codebook``,
  ``capture_run_context``, ``attach_provenance``, ``is_notebook``,
  ``is_google_colab``, ``write_run_manifest``

Typed analysis result objects also support standardized comparison helpers:
``difference(other)`` and ``effect(other)``, plus operator shorthands
``left - right`` and ``left / right``.

.. automodule:: design_research_analysis
   :members:
   :undoc-members:
