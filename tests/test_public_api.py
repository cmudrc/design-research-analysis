"""Tests for the curated public API."""

from __future__ import annotations

import numpy as np
import pytest

import design_research_analysis as package


def test_public_exports_match_the_curated_api() -> None:
    """Keep the top-level exports explicit and stable."""

    assert package.__all__ == [
        "ComparisonResult",
        "DecodeResult",
        "DiscreteHMMResult",
        "EmbeddingMapResult",
        "EmbeddingResult",
        "GaussianHMMResult",
        "MarkovChainResult",
        "UnifiedTableConfig",
        "UnifiedTableValidationReport",
        "attach_provenance",
        "bootstrap_ci",
        "build_condition_metric_table",
        "build_embedding_map",
        "capture_run_context",
        "cluster_embedding_map",
        "coerce_unified_table",
        "compare_condition_pairs",
        "compare_embedding_maps",
        "compare_groups",
        "compute_design_space_coverage",
        "compute_divergence_convergence",
        "compute_idea_space_trajectory",
        "compute_language_convergence",
        "compute_semantic_distance_trajectory",
        "dataset",
        "decode_hmm",
        "derive_columns",
        "embed_records",
        "embedding_maps",
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
        "plot_convergence_curve",
        "plot_design_process_timeline",
        "plot_embedding_map",
        "plot_embedding_map_grid",
        "plot_idea_trajectory",
        "plot_state_graph",
        "plot_transition_matrix",
        "power_curve",
        "profile_dataframe",
        "rank_tests_one_stop",
        "runtime",
        "score_sentiment",
        "sequence",
        "stats",
        "validate_dataframe",
        "validate_unified_table",
        "visualization",
        "write_run_manifest",
    ]


def test_dimred_compatibility_wrappers_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    import design_research_analysis.dimred as legacy_dimred

    projection = legacy_dimred.ProjectionResult(
        coordinates=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        record_ids=["r1", "r2"],
        method="pca",
    )
    cluster_payload = {"labels": [0, 1], "method": "kmeans", "centers": [[0.0, 1.0], [1.0, 0.0]]}
    captured: dict[str, object] = {}

    def _fake_build(embeddings: object, **kwargs: object) -> legacy_dimred.EmbeddingMapResult:
        captured["embeddings"] = np.asarray(embeddings)
        captured["build_kwargs"] = kwargs
        return projection

    def _fake_cluster(
        embedding_map: object,
        **kwargs: object,
    ) -> dict[str, object]:
        captured["projection"] = embedding_map
        captured["cluster_kwargs"] = kwargs
        return cluster_payload

    monkeypatch.setattr(legacy_dimred, "build_embedding_map", _fake_build)
    monkeypatch.setattr(legacy_dimred, "cluster_embedding_map", _fake_cluster)

    reduced = legacy_dimred.reduce_dimensions(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        method="umap",
        n_components=2,
        random_state=17,
        perplexity=9.0,
        n_neighbors=5,
        min_dist=0.2,
    )
    clustered = legacy_dimred.cluster_projection(
        projection,
        method="agglomerative",
        n_clusters=2,
        random_state=3,
        max_iter=25,
    )

    assert reduced is projection
    assert clustered == cluster_payload
    assert np.array_equal(
        captured["embeddings"],
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float),
    )
    assert captured["build_kwargs"] == {
        "method": "umap",
        "n_components": 2,
        "random_state": 17,
        "perplexity": 9.0,
        "n_neighbors": 5,
        "min_dist": 0.2,
    }
    assert captured["projection"] is projection
    assert captured["cluster_kwargs"] == {
        "method": "agglomerative",
        "n_clusters": 2,
        "random_state": 3,
        "max_iter": 25,
    }
    assert legacy_dimred.ProjectionResult is legacy_dimred.EmbeddingMapResult
    assert "reduce_dimensions" in legacy_dimred.__all__
    assert "cluster_projection" in legacy_dimred.__all__
