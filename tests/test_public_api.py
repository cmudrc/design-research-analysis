"""Tests for the curated public API."""

from __future__ import annotations

import design_research_analysis as package


def test_public_exports_match_the_curated_api() -> None:
    """Keep the top-level exports explicit and stable."""

    assert package.__all__ == [
        "DecodeResult",
        "DiscreteHMMResult",
        "GaussianHMMResult",
        "MarkovChainResult",
        "UnifiedTableConfig",
        "UnifiedTableValidationReport",
        "attach_provenance",
        "bootstrap_ci",
        "capture_run_context",
        "cluster_projection",
        "coerce_unified_table",
        "compare_groups",
        "compute_language_convergence",
        "compute_semantic_distance_trajectory",
        "dataset",
        "decode_hmm",
        "derive_columns",
        "dimred",
        "embed_records",
        "estimate_sample_size",
        "fit_discrete_hmm_from_table",
        "fit_markov_chain_from_table",
        "fit_mixed_effects",
        "fit_regression",
        "fit_text_gaussian_hmm_from_table",
        "fit_topic_model",
        "generate_codebook",
        "is_google_colab",
        "is_notebook",
        "language",
        "minimum_detectable_effect",
        "permutation_test",
        "plot_state_graph",
        "plot_transition_matrix",
        "power_curve",
        "profile_dataframe",
        "rank_tests_one_stop",
        "reduce_dimensions",
        "runtime",
        "score_sentiment",
        "sequence",
        "stats",
        "validate_dataframe",
        "validate_unified_table",
        "write_run_manifest",
    ]
