"""Tests for the curated public API."""

from __future__ import annotations

import design_research_analysis as package


def test_public_exports_match_the_curated_api() -> None:
    """Keep the top-level exports explicit and stable."""

    assert package.__all__ == [
        "DecodeResult",
        "DiscreteHMMResult",
        "EmbeddingResult",
        "GaussianHMMResult",
        "GroupComparisonResult",
        "LanguageConvergenceResult",
        "MarkovChainResult",
        "MapperFn",
        "MixedEffectsResult",
        "ProjectBlueprint",
        "ProjectionResult",
        "RegressionResult",
        "UnifiedTableConfig",
        "UnifiedTableValidationReport",
        "build_default_blueprint",
        "cluster_projection",
        "coerce_unified_table",
        "compare_groups",
        "compute_language_convergence",
        "compute_semantic_distance_trajectory",
        "decode_hmm",
        "derive_columns",
        "dimred",
        "embed_records",
        "describe_project",
        "embed_text",
        "fit_discrete_hmm",
        "fit_discrete_hmm_from_table",
        "fit_gaussian_hmm",
        "fit_mixed_effects",
        "fit_markov_chain",
        "fit_markov_chain_from_table",
        "fit_regression",
        "fit_topic_model",
        "fit_text_gaussian_hmm",
        "fit_text_gaussian_hmm_from_table",
        "group_rows",
        "language",
        "normalize_package_name",
        "plot_state_graph",
        "plot_transition_matrix",
        "reduce_dimensions",
        "score_sentiment",
        "select_column",
        "sequence",
        "stats",
        "validate_unified_table",
    ]
