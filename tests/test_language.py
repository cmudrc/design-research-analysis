from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from design_research_analysis.language import (
    compute_language_convergence,
    compute_semantic_distance_trajectory,
    fit_topic_model,
    score_sentiment,
)


def test_compute_language_convergence_detects_converging_trend() -> None:
    rows = [
        {"timestamp": "2026-01-01T10:00:00Z", "session_id": "s1", "text": "far"},
        {"timestamp": "2026-01-01T10:00:01Z", "session_id": "s1", "text": "mid"},
        {"timestamp": "2026-01-01T10:00:02Z", "session_id": "s1", "text": "close"},
        {"timestamp": "2026-01-01T10:00:03Z", "session_id": "s1", "text": "target"},
    ]

    lookup = {
        "far": [0.0, 1.0],
        "mid": [0.4, 0.9],
        "close": [0.8, 0.4],
        "target": [1.0, 0.0],
    }

    def _embed(texts: list[str]) -> list[list[float]]:
        return [lookup[text] for text in texts]

    result = compute_language_convergence(rows, window_size=1, embedder=_embed)

    assert result.direction_by_group["s1"] == "converging"
    assert result.slope_by_group["s1"] < 0.0


def test_compute_language_convergence_handles_text_list_and_stable_trend() -> None:
    """Text-list inputs should use the all-group path and stable slope branch."""

    def _embed(texts: list[str]) -> np.ndarray:
        assert texts == ["alpha", "beta"]
        return np.asarray([[1.0, 0.0], [1.0, 0.0]])

    result = compute_language_convergence(
        ["alpha", "beta"],
        window_size=1,
        embedder=_embed,
    )

    assert result.groups == ["__all__"]
    assert result.direction_by_group["__all__"] == "stable"
    assert result.n_observations == 2
    assert result.to_dict()["config"]["model_name"] == "custom"


def test_compute_language_convergence_detects_diverging_trend() -> None:
    """Positive semantic-distance slope should be labeled as diverging."""
    rows = [
        {"timestamp": "2026-01-01T10:00:00Z", "session_id": "s1", "text": "target-a"},
        {"timestamp": "2026-01-01T10:00:01Z", "session_id": "s1", "text": "target-b"},
        {"timestamp": "2026-01-01T10:00:02Z", "session_id": "s1", "text": "opposite"},
        {"timestamp": "2026-01-01T10:00:03Z", "session_id": "s1", "text": "target-c"},
    ]
    lookup = {
        "target-a": [1.0, 0.0],
        "target-b": [1.0, 0.0],
        "opposite": [-1.0, 0.0],
        "target-c": [1.0, 0.0],
    }

    result = compute_language_convergence(
        rows,
        window_size=1,
        embedder=lambda texts: [lookup[text] for text in texts],
    )

    assert result.direction_by_group["s1"] == "diverging"
    assert result.slope_by_group["s1"] > 0.0


def test_semantic_distance_trajectory_validates_inputs() -> None:
    """Semantic trajectories should reject invalid window and embedder shapes."""
    rows = [{"timestamp": "2026-01-01T10:00:00Z", "session_id": "s1", "text": "alpha"}]

    with pytest.raises(ValueError, match="window_size"):
        compute_semantic_distance_trajectory(rows, window_size=0, embedder=lambda texts: [[1.0]])

    with pytest.raises(ValueError, match="2D embedding"):
        compute_semantic_distance_trajectory(rows, embedder=lambda texts: [1.0])

    with pytest.raises(ValueError, match="rows must match"):
        compute_semantic_distance_trajectory(rows, embedder=lambda texts: [[1.0], [2.0]])


def test_language_text_extraction_handles_mapper_and_blank_values() -> None:
    """Table inputs should support derived text and raise on unresolved blanks."""
    rows = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "session_id": "",
            "event_type": "idea",
            "description": "derived text",
        }
    ]

    trajectories = compute_semantic_distance_trajectory(
        rows,
        window_size=2,
        embedder=lambda texts: [[0.0, 0.0]],
        text_mapper=lambda row: row["description"],
    )
    assert trajectories == {"__all__": []}

    with pytest.raises(ValueError, match="missing text"):
        compute_semantic_distance_trajectory(
            [{"timestamp": "2026-01-01T10:00:00Z", "text": ""}],
            embedder=lambda texts: [[1.0]],
        )


def test_score_sentiment_labels_expected_polarity() -> None:
    rows = [
        {"timestamp": "2026-01-01T10:00:00Z", "text": "great clear success"},
        {"timestamp": "2026-01-01T10:00:01Z", "text": "confusing bad problem"},
    ]
    result = score_sentiment(rows)

    assert result["labels"][0] == "positive"
    assert result["labels"][1] == "negative"


def test_score_sentiment_handles_neutral_and_empty_inputs() -> None:
    """Sentiment scoring should keep empty-token and empty-input paths deterministic."""
    neutral = score_sentiment(["12345", "balanced words"])
    assert neutral["labels"] == ["neutral", "neutral"]
    assert neutral["counts"]["neutral"] == 2

    empty = score_sentiment([])
    assert empty["n_documents"] == 0
    assert empty["mean_score"] == 0.0
    assert empty["std_score"] == 0.0


def test_fit_topic_model_validates_basic_parameters() -> None:
    """Topic modeling should reject invalid lightweight arguments before optional imports."""
    with pytest.raises(ValueError, match="n_topics"):
        fit_topic_model(["alpha"], n_topics=0)

    with pytest.raises(ValueError, match="max_features"):
        fit_topic_model(["alpha"], max_features=0)

    with pytest.raises(ValueError, match="top_k_terms"):
        fit_topic_model(["alpha"], top_k_terms=0)


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="sklearn unavailable")
def test_fit_topic_model_is_deterministic_with_fixed_seed() -> None:
    texts = [
        "prototype testing improved communication with users",
        "communication quality improved after prototype testing",
        "manufacturing process optimization reduced defects",
        "optimization of manufacturing reduced process defects",
    ]

    first = fit_topic_model(texts, n_topics=2, random_state=4)
    second = fit_topic_model(texts, n_topics=2, random_state=4)

    assert first["topic_terms"] == second["topic_terms"]
