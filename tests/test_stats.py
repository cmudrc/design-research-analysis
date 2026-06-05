from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from design_research_analysis.stats import (
    GroupComparisonResult,
    MixedEffectsResult,
    compare_groups,
    fit_mixed_effects,
    fit_regression,
)


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="scipy unavailable")
def test_compare_groups_returns_expected_directionality() -> None:
    values = [1.0, 1.2, 0.9, 5.0, 5.1, 5.2]
    groups = ["A", "A", "A", "B", "B", "B"]

    result = compare_groups(values, groups, method="ttest")

    assert result.method == "ttest"
    assert result.group_means["B"] > result.group_means["A"]
    assert result.p_value < 0.05


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="scipy unavailable")
def test_compare_groups_table_anova_kruskal_and_edge_methods() -> None:
    rows = [
        {"group": "A", "value": 1.0},
        {"group": "A", "value": 1.1},
        {"group": "B", "value": 2.0},
        {"group": "B", "value": 2.1},
        {"group": "C", "value": 3.0},
        {"group": "C", "value": 3.1},
    ]

    anova = compare_groups(data=rows)
    kruskal = compare_groups(data=rows, method="kruskal")

    assert anova.method == "anova"
    assert kruskal.method == "kruskal"
    assert anova.effect_size > 0.0

    constant = compare_groups([1.0, 1.0, 1.0, 1.0], ["A", "A", "B", "B"], method="ttest")
    assert constant.effect_size == 0.0

    singleton = compare_groups([1.0, 2.0], ["A", "B"], method="ttest")
    assert singleton.effect_size == 0.0

    flat = compare_groups([1.0, 1.0, 1.0, 1.0], ["A", "A", "B", "B"], method="anova")
    assert flat.effect_size == 0.0

    with pytest.raises(ValueError, match="Row 0 is missing"):
        compare_groups(data=[{"group": "A"}])
    with pytest.raises(ValueError, match="ttest requires exactly two groups"):
        compare_groups(data=rows, method="ttest")
    with pytest.raises(ValueError, match="Unsupported method"):
        compare_groups(data=rows, method="bogus")


def test_group_and_mixed_results_compare_on_aligned_fields() -> None:
    left = GroupComparisonResult(
        method="ttest",
        statistic=2.0,
        p_value=0.05,
        effect_size=0.4,
        group_means={"A": 1.0},
        group_sizes={"A": 3},
    )
    right = GroupComparisonResult(
        method="anova",
        statistic=3.0,
        p_value=0.01,
        effect_size=0.7,
        group_means={"B": 2.0},
        group_sizes={"B": 4},
    )

    diff = left - right
    assert diff.metric == "group_summary"
    assert diff.details["group_labels"] == ["A", "B"]
    assert diff.details["methods"] == ["ttest", "anova"]

    mixed_left = MixedEffectsResult(
        success=True,
        backend="statsmodels",
        formula="y ~ x",
        group_column="subject",
        params={"Intercept": 1.0},
        aic=None,
        bic=4.0,
        log_likelihood=-1.0,
    )
    mixed_right = MixedEffectsResult(
        success=False,
        backend="statsmodels",
        formula="y ~ z",
        group_column="cohort",
        params={"slope": 2.0},
        aic=5.0,
        bic=None,
        log_likelihood=None,
    )

    mixed_diff = mixed_left - mixed_right
    assert mixed_diff.metric == "mixed_effects_profile"
    assert mixed_diff.details["parameter_names"] == ["Intercept", "slope"]


def test_fit_regression_recovers_linear_relationship() -> None:
    x = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.asarray([1.0, 3.0, 5.0, 7.0, 9.0])

    result = fit_regression(x, y, feature_names=["feature"])

    assert pytest.approx(result.intercept, abs=1e-8) == 1.0
    assert pytest.approx(result.coefficients["feature"], abs=1e-8) == 2.0
    assert pytest.approx(result.r2, abs=1e-8) == 1.0


def test_fit_regression_input_validation() -> None:
    with pytest.raises(ValueError, match="X must be a 2D matrix"):
        fit_regression([1.0, 2.0], [1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="y must be a 1D vector"):
        fit_regression([[1.0], [2.0]], [[1.0], [2.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same number of rows"):
        fit_regression([[1.0], [2.0]], [1.0])
    with pytest.raises(ValueError, match="must not be empty"):
        fit_regression(np.empty((0, 1)), np.empty((0,)))
    with pytest.raises(ValueError, match="feature_names length"):
        fit_regression([[1.0], [2.0]], [1.0, 2.0], feature_names=["a", "b"])


def test_fit_regression_without_intercept_uses_default_feature_names() -> None:
    result = fit_regression(
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
        [3.0, 6.0, 9.0],
        add_intercept=False,
    )

    assert result.intercept == 0.0
    assert set(result.coefficients) == {"x0", "x1"}
    assert result.config == {"add_intercept": False}


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


def test_compare_groups_input_validation_errors() -> None:
    with pytest.raises(ValueError, match="Provide either values/groups"):
        compare_groups()
    with pytest.raises(ValueError, match="same length"):
        compare_groups([1.0, 2.0], ["A"])
    with pytest.raises(ValueError, match="must not be empty"):
        compare_groups([], [])
    with pytest.raises(ValueError, match="At least two groups"):
        compare_groups([1.0, 2.0], ["A", "A"])


def test_fit_mixed_effects_validation_errors() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        fit_mixed_effects([], formula="y ~ x", group_column="g", backend="bad")
    with pytest.raises(ValueError, match="requires at least one row"):
        fit_mixed_effects([], formula="y ~ x", group_column="g")
    with pytest.raises(ValueError, match="group_column 'missing'"):
        fit_mixed_effects([{"g": "a", "y": 1.0, "x": 0.0}], formula="y ~ x", group_column="missing")
