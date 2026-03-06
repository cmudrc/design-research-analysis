from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from design_research_analysis.stats import compare_groups, fit_mixed_effects, fit_regression


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="scipy unavailable")
def test_compare_groups_returns_expected_directionality() -> None:
    values = [1.0, 1.2, 0.9, 5.0, 5.1, 5.2]
    groups = ["A", "A", "A", "B", "B", "B"]

    result = compare_groups(values, groups, method="ttest")

    assert result.method == "ttest"
    assert result.group_means["B"] > result.group_means["A"]
    assert result.p_value < 0.05


def test_fit_regression_recovers_linear_relationship() -> None:
    x = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.asarray([1.0, 3.0, 5.0, 7.0, 9.0])

    result = fit_regression(x, y, feature_names=["feature"])

    assert pytest.approx(result.intercept, abs=1e-8) == 1.0
    assert pytest.approx(result.coefficients["feature"], abs=1e-8) == 2.0
    assert pytest.approx(result.r2, abs=1e-8) == 1.0


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None or importlib.util.find_spec("pandas") is None,
    reason="statsmodels/pandas unavailable",
)
def test_fit_mixed_effects_runs_and_returns_structured_result() -> None:
    rows = [
        {"timestamp": "2026-01-01T10:00:00Z", "subject": "u1", "condition": 0, "outcome": 1.0},
        {"timestamp": "2026-01-01T10:00:01Z", "subject": "u1", "condition": 1, "outcome": 2.0},
        {"timestamp": "2026-01-01T10:00:02Z", "subject": "u2", "condition": 0, "outcome": 1.5},
        {"timestamp": "2026-01-01T10:00:03Z", "subject": "u2", "condition": 1, "outcome": 2.4},
        {"timestamp": "2026-01-01T10:00:04Z", "subject": "u3", "condition": 0, "outcome": 0.9},
        {"timestamp": "2026-01-01T10:00:05Z", "subject": "u3", "condition": 1, "outcome": 2.1},
    ]

    result = fit_mixed_effects(rows, formula="outcome ~ condition", group_column="subject")

    assert isinstance(result.success, bool)
    assert result.backend == "statsmodels"
