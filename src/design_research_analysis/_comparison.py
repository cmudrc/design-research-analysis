"""Internal helpers for algebraic comparison across typed result objects."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ComparisonResult:
    """Structured output for algebraic result-object comparisons."""

    operation: str
    left_type: str
    right_type: str
    metric: str
    estimate: float
    statistic: float | None = None
    p_value: float | None = None
    effect_size: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert the comparison output to a JSON-serializable dictionary."""
        return {
            "operation": self.operation,
            "left_type": self.left_type,
            "right_type": self.right_type,
            "metric": self.metric,
            "estimate": float(self.estimate),
            "statistic": None if self.statistic is None else float(self.statistic),
            "p_value": None if self.p_value is None else float(self.p_value),
            "effect_size": None if self.effect_size is None else float(self.effect_size),
            "details": dict(self.details),
            "interpretation": self.interpretation,
        }


def flatten_numeric_vector(values: Any, *, name: str) -> np.ndarray:
    """Normalize numeric inputs to a non-empty 1D float array."""
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must contain at least one numeric value.")
    return vector


def cohen_d(left: Any, right: Any) -> float:
    """Estimate a standardized mean difference between numeric vectors."""
    left_vec = flatten_numeric_vector(left, name="left")
    right_vec = flatten_numeric_vector(right, name="right")

    if left_vec.size < 2 or right_vec.size < 2:
        return 0.0

    left_var = float(np.var(left_vec, ddof=1))
    right_var = float(np.var(right_vec, ddof=1))
    pooled = (((left_vec.size - 1) * left_var) + ((right_vec.size - 1) * right_var)) / float(
        left_vec.size + right_vec.size - 2
    )
    if pooled <= 0.0:
        return 0.0
    return float((np.mean(left_vec) - np.mean(right_vec)) / math.sqrt(pooled))


def rms_delta(left: Any, right: Any) -> float:
    """Return the root-mean-square delta between paired numeric vectors."""
    left_vec = flatten_numeric_vector(left, name="left")
    right_vec = flatten_numeric_vector(right, name="right")
    if left_vec.shape != right_vec.shape:
        raise ValueError(
            f"Numeric comparison requires matching vector sizes. Got {left_vec.shape} and "
            f"{right_vec.shape}."
        )
    return float(math.sqrt(float(np.mean((left_vec - right_vec) ** 2))))


def permutation_rms_test(
    left: Any,
    right: Any,
    *,
    n_permutations: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Estimate a permutation p-value for RMS difference between vectors."""
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")

    left_vec = flatten_numeric_vector(left, name="left")
    right_vec = flatten_numeric_vector(right, name="right")
    if left_vec.shape != right_vec.shape:
        raise ValueError(
            f"Numeric comparison requires matching vector sizes. Got {left_vec.shape} and "
            f"{right_vec.shape}."
        )

    observed = rms_delta(left_vec, right_vec)
    pooled = np.concatenate([left_vec, right_vec])
    n_left = left_vec.size
    rng = np.random.default_rng(seed)
    exceedances = 0

    for _ in range(n_permutations):
        permuted = rng.permutation(pooled)
        perm_stat = rms_delta(permuted[:n_left], permuted[n_left:])
        if perm_stat >= observed:
            exceedances += 1

    p_value = float((exceedances + 1) / (n_permutations + 1))
    return observed, p_value


def build_numeric_difference_result(
    *,
    left: Any,
    right: Any,
    left_type: str,
    right_type: str,
    metric: str,
    details: dict[str, Any] | None = None,
    seed: int = 0,
) -> ComparisonResult:
    """Build a default difference result from aligned numeric vectors."""
    left_vec = flatten_numeric_vector(left, name="left")
    right_vec = flatten_numeric_vector(right, name="right")
    statistic, p_value = permutation_rms_test(left_vec, right_vec, seed=seed)
    effect = cohen_d(left_vec, right_vec)
    payload = dict(details or {})
    payload.setdefault("n_parameters", int(left_vec.size))
    payload.setdefault("mean_absolute_difference", float(np.mean(np.abs(left_vec - right_vec))))
    interpretation = (
        f"RMS {metric} difference is {statistic:.4g}. "
        f"Permutation p={p_value:.4g}. "
        f"Standardized effect size d={effect:.4g}."
    )
    return ComparisonResult(
        operation="difference",
        left_type=left_type,
        right_type=right_type,
        metric=metric,
        estimate=float(statistic),
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=float(effect),
        details=payload,
        interpretation=interpretation,
    )


def build_numeric_effect_size_result(
    *,
    left: Any,
    right: Any,
    left_type: str,
    right_type: str,
    metric: str,
    details: dict[str, Any] | None = None,
) -> ComparisonResult:
    """Build a default effect-size result from aligned numeric vectors."""
    left_vec = flatten_numeric_vector(left, name="left")
    right_vec = flatten_numeric_vector(right, name="right")
    effect = cohen_d(left_vec, right_vec)
    payload = dict(details or {})
    payload.setdefault("n_parameters", int(left_vec.size))
    payload.setdefault("mean_left", float(np.mean(left_vec)))
    payload.setdefault("mean_right", float(np.mean(right_vec)))
    interpretation = (
        f"Standardized {metric} effect size is d={effect:.4g}. "
        "Positive values indicate larger average parameters on the left-hand result."
    )
    return ComparisonResult(
        operation="effect_size",
        left_type=left_type,
        right_type=right_type,
        metric=metric,
        estimate=float(effect),
        statistic=None,
        p_value=None,
        effect_size=float(effect),
        details=payload,
        interpretation=interpretation,
    )


def align_vector_by_labels(
    values: Any,
    source_labels: list[str],
    target_labels: list[str],
) -> np.ndarray:
    """Expand a 1D vector to a shared label space."""
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != len(source_labels):
        raise ValueError("Label alignment requires one value per source label.")

    index_map = {label: idx for idx, label in enumerate(source_labels)}
    aligned = np.zeros(len(target_labels), dtype=float)
    for idx, label in enumerate(target_labels):
        source_idx = index_map.get(label)
        if source_idx is not None:
            aligned[idx] = vector[source_idx]
    return aligned


def align_square_matrix_by_labels(
    matrix: Any,
    source_labels: list[str],
    target_labels: list[str],
) -> np.ndarray:
    """Expand a square matrix to a shared row/column label space."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Square-matrix alignment requires a square 2D array.")
    if arr.shape[0] != len(source_labels):
        raise ValueError("Label alignment requires one row/column per source label.")

    index_map = {label: idx for idx, label in enumerate(source_labels)}
    aligned = np.zeros((len(target_labels), len(target_labels)), dtype=float)
    for row_idx, row_label in enumerate(target_labels):
        source_row = index_map.get(row_label)
        if source_row is None:
            continue
        for col_idx, col_label in enumerate(target_labels):
            source_col = index_map.get(col_label)
            if source_col is None:
                continue
            aligned[row_idx, col_idx] = arr[source_row, source_col]
    return aligned


def permute_vector(values: Any, permutation: tuple[int, ...]) -> np.ndarray:
    """Reorder a 1D vector according to a state permutation."""
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != len(permutation):
        raise ValueError("Permutation size must match vector length.")
    return vector[list(permutation)]


def permute_rows(matrix: Any, permutation: tuple[int, ...]) -> np.ndarray:
    """Reorder the rows of a matrix according to a state permutation."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != len(permutation):
        raise ValueError("Permutation size must match the number of matrix rows.")
    return arr[list(permutation), :]


def permute_square_matrix(matrix: Any, permutation: tuple[int, ...]) -> np.ndarray:
    """Reorder both axes of a square matrix according to a state permutation."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Square-matrix permutation requires a square 2D array.")
    if arr.shape[0] != len(permutation):
        raise ValueError("Permutation size must match the square matrix dimension.")
    indices = list(permutation)
    return arr[np.ix_(indices, indices)]


def best_assignment(cost_matrix: Any) -> tuple[int, ...]:
    """Find a low-cost one-to-one assignment for a square cost matrix."""
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("Assignment requires a square cost matrix.")
    n_states = cost.shape[0]
    if n_states == 0:
        return ()

    if n_states <= 8:
        best = min(
            itertools.permutations(range(n_states)),
            key=lambda perm: float(sum(cost[idx, perm[idx]] for idx in range(n_states))),
        )
        return tuple(int(item) for item in best)

    remaining = set(range(n_states))
    ordered: list[int] = []
    for row_idx in range(n_states):
        best_col = min(remaining, key=lambda col_idx: float(cost[row_idx, col_idx]))
        ordered.append(int(best_col))
        remaining.remove(best_col)
    return tuple(ordered)


class ComparableResultMixin:
    """Provide shared difference/effect-size operators for result objects."""

    def __sub__(self, other: Any) -> ComparisonResult:
        """Shorthand for ``difference(other)``."""
        return self.difference(other)

    def __truediv__(self, other: Any) -> ComparisonResult:
        """Shorthand for ``effect(other)``."""
        return self.effect(other)

    def difference(self, other: Any) -> ComparisonResult:
        """Return a structured difference result against another typed result."""
        return self._comparison_result(other, operation="difference")

    def effect(self, other: Any) -> ComparisonResult:
        """Return a structured effect-size result against another typed result."""
        return self._comparison_result(other, operation="effect_size")

    def _comparison_result(self, other: Any, *, operation: str) -> ComparisonResult:
        other_family = getattr(other, "_comparison_family", None)
        if other_family is None or other_family() != self._comparison_family():
            raise TypeError(
                f"{type(self).__name__} can only be compared against results in the "
                f"'{self._comparison_family()}' family."
            )
        return self._build_comparison(other, operation=operation)

    def _comparison_family(self) -> str:
        return self.__class__.__name__

    def _comparison_metric(self) -> str:
        return "parameter_profile"

    def _build_comparison(self, other: Any, *, operation: str) -> ComparisonResult:
        left_vector, right_vector, details = self._comparison_vectors(other)
        if operation == "difference":
            return build_numeric_difference_result(
                left=left_vector,
                right=right_vector,
                left_type=type(self).__name__,
                right_type=type(other).__name__,
                metric=self._comparison_metric(),
                details=details,
            )
        if operation == "effect_size":
            return build_numeric_effect_size_result(
                left=left_vector,
                right=right_vector,
                left_type=type(self).__name__,
                right_type=type(other).__name__,
                metric=self._comparison_metric(),
                details=details,
            )
        raise ValueError(f"Unsupported comparison operation: {operation}")

    def _comparison_vectors(self, other: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        raise NotImplementedError
