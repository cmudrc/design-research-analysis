from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from design_research_analysis._comparison import (
    ComparableResultMixin,
    ComparisonResult,
    align_square_matrix_by_labels,
    align_vector_by_labels,
    best_assignment,
    build_numeric_difference_result,
    build_numeric_effect_size_result,
    cohen_d,
    flatten_numeric_vector,
    permutation_rms_test,
    permute_rows,
    permute_square_matrix,
    permute_vector,
    rms_delta,
)


def test_comparison_result_and_numeric_helper_edge_cases() -> None:
    result = ComparisonResult(
        operation="difference",
        left_type="Left",
        right_type="Right",
        metric="metric",
        estimate=1,
        statistic=2,
        p_value=0.25,
        effect_size=0.5,
        details={"n": 2},
        interpretation="done",
    )

    assert result.to_dict()["estimate"] == 1.0
    assert np.array_equal(flatten_numeric_vector([[1, 2]], name="values"), np.asarray([1.0, 2.0]))

    with pytest.raises(ValueError, match="must contain at least one numeric value"):
        flatten_numeric_vector([], name="values")

    assert cohen_d([1.0], [2.0]) == 0.0
    assert cohen_d([1.0, 1.0], [1.0, 1.0]) == 0.0


def test_rms_and_permutation_helpers_validate_shapes_and_counts() -> None:
    with pytest.raises(ValueError, match="matching vector sizes"):
        rms_delta([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="must be positive"):
        permutation_rms_test([1.0], [1.0], n_permutations=0)
    with pytest.raises(ValueError, match="matching vector sizes"):
        permutation_rms_test([1.0, 2.0], [1.0], n_permutations=5)

    observed, p_value = permutation_rms_test([1.0, 2.0], [1.0, 2.0], n_permutations=5, seed=3)
    assert observed == pytest.approx(0.0)
    assert 0.0 <= p_value <= 1.0


def test_build_numeric_results_and_alignment_helpers() -> None:
    difference = build_numeric_difference_result(
        left=[1.0, 2.0],
        right=[1.5, 2.5],
        left_type="Left",
        right_type="Right",
        metric="profile",
    )
    effect = build_numeric_effect_size_result(
        left=[1.0, 2.0],
        right=[1.5, 2.5],
        left_type="Left",
        right_type="Right",
        metric="profile",
    )

    assert difference.details["n_parameters"] == 2
    assert effect.details["mean_left"] == 1.5

    assert np.array_equal(
        align_vector_by_labels([1.0, 2.0], ["a", "b"], ["b", "c", "a"]),
        np.asarray([2.0, 0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="one value per source label"):
        align_vector_by_labels([1.0], ["a", "b"], ["a"])

    assert np.array_equal(
        align_square_matrix_by_labels(
            [[1.0, 2.0], [3.0, 4.0]],
            ["a", "b"],
            ["b", "c", "a"],
        ),
        np.asarray([[4.0, 0.0, 3.0], [0.0, 0.0, 0.0], [2.0, 0.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="square 2D array"):
        align_square_matrix_by_labels([1.0, 2.0], ["a"], ["a"])
    with pytest.raises(ValueError, match="one row/column per source label"):
        align_square_matrix_by_labels([[1.0]], ["a", "b"], ["a"])


def test_permutation_helpers_and_best_assignment_paths() -> None:
    assert np.array_equal(permute_vector([1.0, 2.0], (1, 0)), np.asarray([2.0, 1.0]))
    with pytest.raises(ValueError, match="vector length"):
        permute_vector([1.0], (0, 1))

    assert np.array_equal(
        permute_rows([[1.0, 2.0], [3.0, 4.0]], (1, 0)),
        np.asarray([[3.0, 4.0], [1.0, 2.0]]),
    )
    with pytest.raises(ValueError, match="number of matrix rows"):
        permute_rows([1.0, 2.0], (0,))

    assert np.array_equal(
        permute_square_matrix([[1.0, 2.0], [3.0, 4.0]], (1, 0)),
        np.asarray([[4.0, 3.0], [2.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="square 2D array"):
        permute_square_matrix([1.0, 2.0], (0,))
    with pytest.raises(ValueError, match="square matrix dimension"):
        permute_square_matrix([[1.0, 2.0], [3.0, 4.0]], (0,))

    with pytest.raises(ValueError, match="square cost matrix"):
        best_assignment([[1.0, 2.0]])
    assert best_assignment(np.empty((0, 0))) == ()
    low_diagonal_cost = np.ones((3, 3)) - np.eye(3)
    large_low_diagonal_cost = np.ones((9, 9)) - np.eye(9)
    assert best_assignment(low_diagonal_cost) == (0, 1, 2)
    assert best_assignment(large_low_diagonal_cost) == (0, 1, 2, 3, 4, 5, 6, 7, 8)


@dataclass
class _ToyComparable(ComparableResultMixin):
    values: np.ndarray

    def _comparison_vectors(
        self,
        other: _ToyComparable,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        return self.values, other.values, {"kind": "toy"}


@dataclass
class _IncompleteComparable(ComparableResultMixin):
    values: np.ndarray


def test_comparable_result_mixin_default_metric_and_error_paths() -> None:
    left = _ToyComparable(np.asarray([1.0, 2.0]))
    right = _ToyComparable(np.asarray([1.0, 3.0]))

    difference = left.difference(right)
    effect = left.effect(right)

    assert difference.metric == "parameter_profile"
    assert effect.operation == "effect_size"

    with pytest.raises(ValueError, match="Unsupported comparison operation"):
        left._build_comparison(right, operation="mystery")
    with pytest.raises(NotImplementedError):
        _IncompleteComparable(np.asarray([1.0]))._comparison_vectors(object())
