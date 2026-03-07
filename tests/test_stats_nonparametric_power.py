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

_HAS_SCIPY = importlib.util.find_spec("scipy") is not None
_HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None
_HAS_PANDAS = importlib.util.find_spec("pandas") is not None


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


@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy unavailable")
def test_rank_tests_one_stop_dispatches_default() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 40)
    y = rng.normal(1.0, 1.0, 40)
    result = rank_tests_one_stop(x, y)
    assert result["test"] == "mannwhitney"
    assert "interpretation" in result


@pytest.mark.parametrize("test_name", ["one_sample_t", "paired_t", "two_sample_t"])
@pytest.mark.skipif(not _HAS_STATSMODELS, reason="statsmodels unavailable")
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


@pytest.mark.skipif(not (_HAS_STATSMODELS and _HAS_PANDAS), reason="statsmodels/pandas unavailable")
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


def test_bootstrap_ci_additional_paths() -> None:
    result = bootstrap_ci([1, 2, 3, 4], stat="median", n_resamples=200, seed=1)
    assert result["method_used"] == "percentile"

    with pytest.raises(ValueError, match="n_resamples must be positive"):
        bootstrap_ci([1, 2], n_resamples=0)
    with pytest.raises(ValueError, match="ci must be in"):
        bootstrap_ci([1, 2], ci=1.0)
    with pytest.raises(ValueError, match="method must be one of"):
        bootstrap_ci([1, 2], method="unknown")
    with pytest.raises(ValueError, match="requires y values"):
        bootstrap_ci([1, 2], stat="diff_means")


@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy unavailable")
def test_bootstrap_bca_and_rank_test_variants() -> None:
    bca = bootstrap_ci([1, 2, 3, 4, 5], stat="mean", method="bca", n_resamples=200, seed=2)
    assert bca["method_used"] == "bca"

    bca_two_sample = bootstrap_ci(
        [1, 2, 3, 4, 5],
        y=[2, 3, 4, 5, 6],
        stat="diff_means",
        method="bca",
        n_resamples=200,
        seed=3,
    )
    assert bca_two_sample["method_used"] == "bca"

    mw = rank_tests_one_stop([1, 2, 3], [3, 4, 5], kind="mannwhitney")
    assert mw["test"] == "mannwhitney"
    wil = rank_tests_one_stop([1, 2, 3], [1.1, 2.1, 3.1], kind="wilcoxon")
    assert wil["test"] == "wilcoxon"
    kr = rank_tests_one_stop([1, 2, 3], groups=[[1, 2], [2, 3], [3, 4]], kind="kruskal")
    assert kr["test"] == "kruskal"
    fr = rank_tests_one_stop([1, 2], groups=[[1, 2], [1, 2], [2, 3]], kind="friedman")
    assert fr["test"] == "friedman"


@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy unavailable")
def test_rank_tests_validation_errors() -> None:
    with pytest.raises(ValueError, match="Provide y or groups"):
        rank_tests_one_stop([1, 2, 3])
    with pytest.raises(ValueError, match="kind must be one of"):
        rank_tests_one_stop([1, 2], [3, 4], kind="bad")
    with pytest.raises(ValueError, match="mannwhitney requires y"):
        rank_tests_one_stop([1, 2], kind="mannwhitney")
    with pytest.raises(ValueError, match="wilcoxon requires y"):
        rank_tests_one_stop([1, 2], kind="wilcoxon")
    with pytest.raises(ValueError, match="kruskal requires groups"):
        rank_tests_one_stop([1, 2], kind="kruskal")
    with pytest.raises(ValueError, match="friedman requires groups"):
        rank_tests_one_stop([1, 2], kind="friedman")


def test_permutation_test_alternative_modes_and_errors() -> None:
    greater = permutation_test(
        [1, 2, 3], [0, 1, 2], alternative="greater", n_permutations=300, seed=0
    )
    less = permutation_test([1, 2, 3], [0, 1, 2], alternative="less", n_permutations=300, seed=0)
    assert 0.0 <= greater["p_value"] <= 1.0
    assert 0.0 <= less["p_value"] <= 1.0
    with pytest.raises(ValueError, match="n_permutations must be positive"):
        permutation_test([1], [2], n_permutations=0)
    with pytest.raises(ValueError, match="alternative must be one of"):
        permutation_test([1], [2], alternative="bad")


class _FakeTwoSamplePower:
    def solve_power(
        self,
        *,
        effect_size: float,
        nobs1: float | None,
        alpha: float,
        power: float,
        ratio: float,
        alternative: str,
    ) -> float:
        _ = (effect_size, nobs1, alpha, power, ratio, alternative)
        return 11.2

    def power(
        self,
        *,
        effect_size: float,
        nobs1: int,
        alpha: float,
        ratio: float,
        alternative: str,
    ) -> float:
        _ = (effect_size, nobs1, alpha, ratio, alternative)
        return 0.82


class _FakeOneSamplePower:
    def solve_power(
        self,
        *,
        effect_size: float,
        nobs: float | None,
        alpha: float,
        power: float,
        alternative: str,
    ) -> float:
        _ = (effect_size, nobs, alpha, power, alternative)
        return 9.7

    def power(
        self,
        *,
        effect_size: float,
        nobs: int,
        alpha: float,
        alternative: str,
    ) -> float:
        _ = (effect_size, nobs, alpha, alternative)
        return 0.84 if effect_size >= 0.3 else 0.6


def test_power_functions_with_fake_engines(monkeypatch) -> None:
    monkeypatch.setattr(
        stats_module,
        "_load_power_engines",
        lambda: (_FakeTwoSamplePower(), _FakeOneSamplePower()),
    )

    two = estimate_sample_size(0.5, test="two_sample_t", ratio=1.5)
    one = estimate_sample_size(0.5, test="paired_t")
    assert two["recommended_n"] > 0
    assert two["group_allocation"] is not None
    assert one["group_allocation"] is None

    if _HAS_PANDAS:
        curve = power_curve([0.2, 0.3], n=24, test="paired_t")
        assert list(curve.columns) == ["effect_size", "power"]
    else:
        with pytest.raises(ImportError, match="Power analysis requires optional dependencies"):
            power_curve([0.2, 0.3], n=24, test="paired_t")

    mde = minimum_detectable_effect(24, test="paired_t", power=0.8)
    assert mde["minimum_detectable_effect"] > 0


def test_power_input_validation_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        stats_module,
        "_load_power_engines",
        lambda: (_FakeTwoSamplePower(), _FakeOneSamplePower()),
    )

    with pytest.raises(ValueError, match="test must be one of"):
        estimate_sample_size(0.5, test="bad")
    with pytest.raises(ValueError, match="alpha must be in"):
        estimate_sample_size(0.5, test="paired_t", alpha=1.0)
    with pytest.raises(ValueError, match="power must be in"):
        estimate_sample_size(0.5, test="paired_t", power=1.0)
    with pytest.raises(ValueError, match="ratio must be positive"):
        estimate_sample_size(0.5, test="two_sample_t", ratio=0.0)
    with pytest.raises(ValueError, match="alternative must be one of"):
        estimate_sample_size(0.5, test="paired_t", alternative="bad")
    with pytest.raises(ValueError, match="effect_size must be non-zero"):
        estimate_sample_size(0.0, test="paired_t")
    with pytest.raises(ValueError, match="effect_sizes must not be empty"):
        power_curve([], n=10, test="paired_t")
    with pytest.raises(ValueError, match="n must be greater than 1"):
        power_curve([0.2], n=1, test="paired_t")
