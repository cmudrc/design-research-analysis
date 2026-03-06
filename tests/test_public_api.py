"""Tests for the curated public API."""

from __future__ import annotations

import design_research_analysis as package


def test_public_exports_match_the_curated_api() -> None:
    """Keep the top-level exports explicit and stable."""

    assert package.__all__ == [
        "UnifiedTableConfig",
        "UnifiedTableValidationReport",
        "cluster_projection",
        "coerce_unified_table",
        "compare_groups",
        "compute_language_convergence",
        "compute_semantic_distance_trajectory",
        "derive_columns",
        "dimred",
        "embed_records",
        "fit_discrete_hmm_from_table",
        "fit_markov_chain_from_table",
        "fit_mixed_effects",
        "fit_regression",
        "fit_text_gaussian_hmm_from_table",
        "fit_topic_model",
        "language",
        "plot_state_graph",
        "plot_transition_matrix",
        "reduce_dimensions",
        "score_sentiment",
        "sequence",
        "stats",
        "validate_unified_table",
    ]
