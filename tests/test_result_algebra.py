from __future__ import annotations

import numpy as np
import pytest

from design_research_analysis import ComparisonResult
from design_research_analysis.language import LanguageConvergenceResult
from design_research_analysis.sequence import (
    DiscreteHMMResult,
    GaussianHMMResult,
    fit_markov_chain,
)
from design_research_analysis.stats import RegressionResult


def test_markov_chain_operators_return_structured_comparison_results() -> None:
    left = fit_markov_chain([["A", "B", "A", "B"], ["A", "A", "B", "A"]], smoothing=0.5)
    right = fit_markov_chain([["A", "A", "A", "B"], ["B", "A", "B", "B"]], smoothing=0.5)

    difference = left - right
    effect = left / right

    assert isinstance(difference, ComparisonResult)
    assert difference.operation == "difference"
    assert difference.metric == "transition_profile"
    assert difference.statistic is not None
    assert difference.effect_size is not None
    assert sorted(difference.details["state_labels"]) == ["A", "B"]
    assert effect.operation == "effect_size"
    assert effect.estimate == effect.effect_size


def test_gaussian_hmm_operator_alignment_handles_state_permutations() -> None:
    left = GaussianHMMResult(
        model=object(),
        n_states=2,
        startprob=np.asarray([0.7, 0.3], dtype=float),
        transmat=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=float),
        means=np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=float),
        covars=np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=float),
    )
    right = GaussianHMMResult(
        model=object(),
        n_states=2,
        startprob=np.asarray([0.3, 0.7], dtype=float),
        transmat=np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=float),
        means=np.asarray([[10.0, 10.0], [0.0, 0.0]], dtype=float),
        covars=np.asarray([[2.0, 2.0], [1.0, 1.0]], dtype=float),
    )

    difference = left - right

    assert difference.estimate == pytest.approx(0.0, abs=1e-12)
    assert difference.details["state_permutation"] == [1, 0]


def test_discrete_hmm_operator_alignment_handles_vocab_and_state_permutations() -> None:
    left = DiscreteHMMResult(
        model=object(),
        n_states=2,
        startprob=np.asarray([0.6, 0.4], dtype=float),
        transmat=np.asarray([[0.75, 0.25], [0.15, 0.85]], dtype=float),
        emissionprob=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=float),
        vocab=["A", "B"],
        token_to_id={"A": 0, "B": 1},
    )
    right = DiscreteHMMResult(
        model=object(),
        n_states=2,
        startprob=np.asarray([0.4, 0.6], dtype=float),
        transmat=np.asarray([[0.85, 0.15], [0.25, 0.75]], dtype=float),
        emissionprob=np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=float),
        vocab=["B", "A"],
        token_to_id={"B": 0, "A": 1},
    )

    difference = left - right

    assert difference.estimate == pytest.approx(0.0, abs=1e-12)
    assert difference.details["state_permutation"] == [1, 0]
    assert difference.details["aligned_vocab"] == ["'A'", "'B'"]


def test_generic_result_objects_support_difference_and_effect_size() -> None:
    left = RegressionResult(
        coefficients={"x": 2.0},
        intercept=1.0,
        r2=0.95,
        mse=0.1,
        n_samples=10,
        n_features=1,
    )
    right = RegressionResult(
        coefficients={"x": 1.0},
        intercept=0.0,
        r2=0.80,
        mse=0.5,
        n_samples=10,
        n_features=1,
    )

    difference = left - right
    effect = left / right

    assert difference.metric == "regression_profile"
    assert difference.p_value is not None
    assert effect.operation == "effect_size"
    assert isinstance(effect.to_dict(), dict)


def test_language_convergence_operators_align_on_group_names() -> None:
    left = LanguageConvergenceResult(
        groups=["s1"],
        distance_trajectories={"s1": [0.4, 0.2]},
        slope_by_group={"s1": -0.2},
        direction_by_group={"s1": "converging"},
        window_size=2,
        n_observations=4,
    )
    right = LanguageConvergenceResult(
        groups=["s2"],
        distance_trajectories={"s2": [0.3, 0.3, 0.3]},
        slope_by_group={"s2": 0.0},
        direction_by_group={"s2": "stable"},
        window_size=2,
        n_observations=4,
    )

    difference = left - right

    assert difference.details["groups"] == ["s1", "s2"]
    assert difference.metric == "convergence_profile"


def test_incompatible_result_families_raise_type_error() -> None:
    markov = fit_markov_chain([["A", "B", "A"]])
    regression = RegressionResult(
        coefficients={"x": 1.0},
        intercept=0.0,
        r2=1.0,
        mse=0.0,
        n_samples=3,
        n_features=1,
    )

    with pytest.raises(TypeError, match="family"):
        _ = markov - regression
