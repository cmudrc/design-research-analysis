"""Walk through a broad slice of the public API with lightweight calls."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from design_research_analysis import (
    DecodeResult,
    DiscreteHMMResult,
    GaussianHMMResult,
    MarkovChainResult,
    UnifiedTableConfig,
    UnifiedTableValidationReport,
    attach_provenance,
    bootstrap_ci,
    capture_run_context,
    cluster_projection,
    coerce_unified_table,
    compare_groups,
    compute_language_convergence,
    compute_semantic_distance_trajectory,
    dataset,
    decode_hmm,
    derive_columns,
    dimred,
    embed_records,
    estimate_sample_size,
    fit_discrete_hmm_from_table,
    fit_markov_chain_from_table,
    fit_mixed_effects,
    fit_regression,
    fit_text_gaussian_hmm_from_table,
    fit_topic_model,
    generate_codebook,
    is_google_colab,
    is_notebook,
    language,
    minimum_detectable_effect,
    permutation_test,
    plot_state_graph,
    plot_transition_matrix,
    power_curve,
    profile_dataframe,
    rank_tests_one_stop,
    reduce_dimensions,
    runtime,
    score_sentiment,
    sequence,
    stats,
    validate_dataframe,
    validate_unified_table,
    write_run_manifest,
)


def _touch_public_api() -> None:
    """Reference optional API symbols without requiring optional deps at runtime."""
    _ = (
        DecodeResult,
        DiscreteHMMResult,
        GaussianHMMResult,
        MarkovChainResult,
        UnifiedTableValidationReport,
        compare_groups,
        dataset,
        decode_hmm,
        dimred,
        embed_records,
        estimate_sample_size,
        fit_discrete_hmm_from_table,
        fit_mixed_effects,
        fit_text_gaussian_hmm_from_table,
        fit_topic_model,
        generate_codebook,
        is_google_colab,
        is_notebook,
        language,
        minimum_detectable_effect,
        plot_state_graph,
        plot_transition_matrix,
        power_curve,
        profile_dataframe,
        rank_tests_one_stop,
        runtime,
        sequence,
        stats,
        validate_dataframe,
    )


def main() -> None:
    """Run a deterministic subset of public API calls that work with base dependencies."""
    rows = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "session_id": "s1",
            "event_type": "A",
            "text": "alpha",
        },
        {
            "timestamp": "2026-01-01T10:00:01Z",
            "session_id": "s1",
            "event_type": "B",
            "text": "beta",
        },
        {
            "timestamp": "2026-01-01T10:00:02Z",
            "session_id": "s2",
            "event_type": "A",
            "text": "gamma",
        },
    ]

    config = UnifiedTableConfig()
    normalized = coerce_unified_table(rows, config=config)
    report = validate_unified_table(normalized, config=config)
    assert report.is_valid
    derived = derive_columns(normalized)

    markov = fit_markov_chain_from_table(derived)

    lookup = {"alpha": [2.0, 0.0], "beta": [1.0, 0.0], "gamma": [0.0, 0.0]}
    trajectory = compute_semantic_distance_trajectory(
        derived,
        window_size=1,
        embedder=lambda texts: [lookup[text] for text in texts],
    )
    convergence = compute_language_convergence(
        derived,
        window_size=1,
        embedder=lambda texts: [lookup[text] for text in texts],
    )
    sentiment = score_sentiment(derived)

    vectors = np.asarray([[1.0, 0.0, 0.1], [0.9, 0.2, 0.1], [0.0, 1.0, 0.2]], dtype=float)
    projection = reduce_dimensions(vectors, method="pca", n_components=2)
    clusters = cluster_projection(projection.projection, n_clusters=2)
    regression = fit_regression([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0], feature_names=["x"])

    bootstrap = bootstrap_ci([1, 2, 3, 4, 5], n_resamples=500, seed=0)
    permutation = permutation_test([1, 2, 3], [3, 4, 5], n_permutations=500, seed=0)

    context = capture_run_context(seed=13)
    manifest = Path("artifacts/runtime/api_surface_manifest.json")
    write_run_manifest(context, manifest)
    payload = attach_provenance({"report_ok": report.is_valid}, context)

    _touch_public_api()

    print(f"Markov states: {len(markov.states)}")
    print(f"Trajectory groups: {sorted(trajectory)}")
    print(f"Convergence labels: {convergence.direction_by_group}")
    print(f"Sentiment docs: {sentiment['n_documents']}")
    print(f"Clusters: {clusters['labels']}")
    print(f"Regression R2: {regression.r2:.3f}")
    print(f"Bootstrap estimate: {bootstrap['estimate']:.3f}")
    print(f"Permutation p-value: {permutation['p_value']:.3f}")
    print(f"Manifest written: {manifest}")
    print(f"Payload keys: {sorted(payload)}")


if __name__ == "__main__":
    main()
