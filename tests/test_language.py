from __future__ import annotations

import importlib.util

import pytest

from design_research_analysis.language import (
    compute_language_convergence,
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
        "far": [4.0, 0.0],
        "mid": [2.0, 0.0],
        "close": [1.0, 0.0],
        "target": [0.0, 0.0],
    }

    def _embed(texts: list[str]) -> list[list[float]]:
        return [lookup[text] for text in texts]

    result = compute_language_convergence(rows, window_size=1, embedder=_embed)

    assert result.direction_by_group["s1"] == "converging"
    assert result.slope_by_group["s1"] < 0.0


def test_score_sentiment_labels_expected_polarity() -> None:
    rows = [
        {"timestamp": "2026-01-01T10:00:00Z", "text": "great clear success"},
        {"timestamp": "2026-01-01T10:00:01Z", "text": "confusing bad problem"},
    ]
    result = score_sentiment(rows)

    assert result["labels"][0] == "positive"
    assert result["labels"][1] == "negative"


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
