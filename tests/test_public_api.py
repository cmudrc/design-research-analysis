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
        "ProjectBlueprint",
        "build_default_blueprint",
        "decode_hmm",
        "describe_project",
        "embed_text",
        "fit_discrete_hmm",
        "fit_gaussian_hmm",
        "fit_markov_chain",
        "fit_text_gaussian_hmm",
        "normalize_package_name",
        "plot_state_graph",
        "plot_transition_matrix",
        "sequence",
    ]
