from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import design_research_analysis.stats as stats_module
from design_research_analysis.stats import (
    bootstrap_ci,
    estimate_sample_size,
    minimum_detectable_effect,
    permutation_test,
    power_curve,
    rank_tests_one_stop,
)


def test_bootstrap_ci_deterministic_bounds() -> None:
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    result = bootstrap_ci(x, stat="mean", n_resamples=4000, seed=2)
    assert 1.5 < result["ci_low"] < 3.2
    assert 3.0 < result["ci_high"] < 4.8


def test_permutation_detects_shift() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, 80)
    y = rng.normal(1.0, 1.0, 80)
    result = permutation_test(x, y, n_permutations=5000, seed=7)
    assert result["p_value"] < 0.05


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="scipy unavailable")
def test_rank_tests_one_stop_dispatches_default() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 40)
    y = rng.normal(1.0, 1.0, 40)
    result = rank_tests_one_stop(x, y)
    assert result["test"] == "mannwhitney"
    assert "interpretation" in result


@pytest.mark.parametrize("test_name", ["one_sample_t", "paired_t", "two_sample_t"])
@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels unavailable"
)
def test_estimate_sample_size_supported_tests(test_name: str) -> None:
    result = estimate_sample_size(0.5, test=test_name)

    assert result["recommended_n"] > 0
    if test_name == "two_sample_t":
        assert result["group_allocation"] is not None
    else:
        assert result["group_allocation"] is None


def test_estimate_sample_size_rejects_zero_effect() -> None:
    with pytest.raises(ValueError):
        estimate_sample_size(0.0, test="paired_t")


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels unavailable"
)
def test_power_curve_and_mde_behave_as_expected() -> None:
    curve = power_curve([0.6, 0.2, 0.4], n=48, test="two_sample_t")
    result = minimum_detectable_effect(48, test="two_sample_t", power=0.8)

    assert list(curve.columns) == ["effect_size", "power"]
    assert list(curve["effect_size"]) == [0.6, 0.2, 0.4]
    assert result["minimum_detectable_effect"] > 0
    achieved = power_curve(
        [result["minimum_detectable_effect"]],
        n=48,
        test="two_sample_t",
    ).iloc[0]["power"]
    assert achieved >= 0.79


def test_power_functions_surface_import_error(monkeypatch) -> None:
    def _raise() -> None:
        raise ImportError(
            "Power analysis requires optional dependencies. "
            "Install with `pip install design-research-analysis[stats]`."
        )

    monkeypatch.setattr(stats_module, "_load_power_engines", _raise)

    with pytest.raises(ImportError, match="Power analysis requires optional dependencies"):
        estimate_sample_size(0.5, test="paired_t")
