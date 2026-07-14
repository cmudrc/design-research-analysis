"""Inter-rater reliability metrics for nominal protocol codings."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_SUPPORTED_METHODS = {
    "cohen_kappa",
    "fleiss_kappa",
    "krippendorff_alpha",
}


@dataclass(frozen=True, slots=True)
class InterraterReliabilityResult:
    """Nominal inter-rater reliability estimate and data-use summary."""

    method: str
    coefficient: float
    n_items: int
    n_items_used: int
    n_raters: int
    n_observations: int
    categories: tuple[Any, ...]
    missing_ratings: int
    confidence_interval: tuple[float, float] | None = None
    bootstrap_samples: int = 0
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "method": self.method,
            "coefficient": float(self.coefficient),
            "n_items": int(self.n_items),
            "n_items_used": int(self.n_items_used),
            "n_raters": int(self.n_raters),
            "n_observations": int(self.n_observations),
            "categories": list(self.categories),
            "missing_ratings": int(self.missing_ratings),
            "confidence_interval": (
                None
                if self.confidence_interval is None
                else [float(value) for value in self.confidence_interval]
            ),
            "bootstrap_samples": int(self.bootstrap_samples),
            "config": dict(self.config),
        }


def _is_missing(value: Any) -> bool:
    """Return whether one nominal rating should be treated as missing."""
    return value is None or (isinstance(value, float | np.floating) and bool(np.isnan(value)))


def _coerce_matrix(codings: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """Validate and copy an item-by-rater nominal coding matrix."""
    if isinstance(codings, str | bytes):
        raise ValueError("codings must be an item-by-rater matrix, not text.")

    matrix: list[list[Any]] = []
    for index, row in enumerate(codings):
        if isinstance(row, str | bytes):
            raise ValueError(f"codings row {index} must be a sequence of ratings, not text.")
        try:
            copied = list(row)
        except TypeError as exc:
            raise ValueError(f"codings row {index} must be a sequence of ratings.") from exc
        matrix.append(copied)

    if len(matrix) < 2:
        raise ValueError("codings must contain at least two items.")
    n_raters = len(matrix[0])
    if n_raters < 2:
        raise ValueError("codings must contain at least two raters.")
    for index, row in enumerate(matrix):
        if len(row) != n_raters:
            raise ValueError(
                f"codings must be rectangular; row {index} has {len(row)} ratings, "
                f"expected {n_raters}."
            )
        for rating in row:
            if _is_missing(rating):
                continue
            try:
                hash(rating)
            except TypeError as exc:
                raise ValueError("Nominal ratings must be hashable values.") from exc
    return matrix


def _category_counts(ratings: Sequence[Any]) -> dict[Any, int]:
    counts: dict[Any, int] = defaultdict(int)
    for rating in ratings:
        counts[rating] += 1
    return dict(counts)


def _sorted_categories(ratings: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(
        sorted(
            set(ratings),
            key=lambda value: (type(value).__qualname__, repr(value)),
        )
    )


def _require_defined(expected_agreement: float) -> None:
    if math.isclose(expected_agreement, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Reliability is undefined because all usable ratings belong to one category."
        )


def _cohen_kappa(matrix: Sequence[Sequence[Any]]) -> tuple[float, int, int, tuple[Any, ...]]:
    if len(matrix[0]) != 2:
        raise ValueError("cohen_kappa requires exactly two raters.")
    complete = [row for row in matrix if not any(_is_missing(value) for value in row)]
    if len(complete) < 2:
        raise ValueError("cohen_kappa requires at least two complete items.")

    first_counts = _category_counts([row[0] for row in complete])
    second_counts = _category_counts([row[1] for row in complete])
    categories = _sorted_categories([value for row in complete for value in row])
    n_items = len(complete)
    observed_agreement = sum(row[0] == row[1] for row in complete) / n_items
    expected_agreement = sum(
        (first_counts.get(category, 0) / n_items) * (second_counts.get(category, 0) / n_items)
        for category in categories
    )
    _require_defined(expected_agreement)
    coefficient = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    return float(coefficient), n_items, n_items * 2, categories


def _fleiss_kappa(matrix: Sequence[Sequence[Any]]) -> tuple[float, int, int, tuple[Any, ...]]:
    complete = [row for row in matrix if not any(_is_missing(value) for value in row)]
    if len(complete) < 2:
        raise ValueError("fleiss_kappa requires at least two complete items.")

    n_items = len(complete)
    n_raters = len(complete[0])
    categories = _sorted_categories([value for row in complete for value in row])
    agreement_by_item: list[float] = []
    total_counts: dict[Any, int] = defaultdict(int)
    for row in complete:
        counts = _category_counts(row)
        for category, count in counts.items():
            total_counts[category] += count
        agreement_by_item.append(
            (sum(count * count for count in counts.values()) - n_raters)
            / (n_raters * (n_raters - 1))
        )

    observed_agreement = float(np.mean(agreement_by_item))
    n_observations = n_items * n_raters
    expected_agreement = sum(
        (total_counts.get(category, 0) / n_observations) ** 2 for category in categories
    )
    _require_defined(expected_agreement)
    coefficient = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    return float(coefficient), n_items, n_observations, categories


def _krippendorff_alpha(
    matrix: Sequence[Sequence[Any]],
) -> tuple[float, int, int, tuple[Any, ...]]:
    usable = [
        [value for value in row if not _is_missing(value)]
        for row in matrix
        if sum(not _is_missing(value) for value in row) >= 2
    ]
    if len(usable) < 2:
        raise ValueError(
            "krippendorff_alpha requires at least two items with two observed ratings."
        )

    observed_disagreement_sum = 0.0
    total_counts: dict[Any, int] = defaultdict(int)
    n_observations = 0
    for row in usable:
        counts = _category_counts(row)
        n_item_ratings = len(row)
        observed_disagreement_sum += (
            n_item_ratings * n_item_ratings - sum(count * count for count in counts.values())
        ) / (n_item_ratings - 1)
        n_observations += n_item_ratings
        for category, count in counts.items():
            total_counts[category] += count

    categories = _sorted_categories(list(total_counts))
    observed_disagreement = observed_disagreement_sum / n_observations
    expected_disagreement = (
        n_observations * n_observations - sum(count * count for count in total_counts.values())
    ) / (n_observations * (n_observations - 1))
    if math.isclose(expected_disagreement, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Reliability is undefined because all usable ratings belong to one category."
        )
    coefficient = 1.0 - (observed_disagreement / expected_disagreement)
    return float(coefficient), len(usable), n_observations, categories


def _coefficient(
    matrix: Sequence[Sequence[Any]],
    *,
    method: str,
) -> tuple[float, int, int, tuple[Any, ...]]:
    if method == "cohen_kappa":
        return _cohen_kappa(matrix)
    if method == "fleiss_kappa":
        return _fleiss_kappa(matrix)
    if method == "krippendorff_alpha":
        return _krippendorff_alpha(matrix)
    valid = ", ".join(sorted(_SUPPORTED_METHODS))
    raise ValueError(f"Unsupported method {method!r}. Valid options: {valid}.")


def _bootstrap_interval(
    matrix: Sequence[Sequence[Any]],
    *,
    method: str,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> tuple[tuple[float, float], int]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    n_items = len(matrix)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_items, size=n_items)
        sample = [matrix[int(index)] for index in indices]
        try:
            estimate, *_ = _coefficient(sample, method=method)
        except ValueError:
            continue
        estimates.append(estimate)

    if len(estimates) < 10:
        raise ValueError(
            "Bootstrap reliability interval produced fewer than 10 valid resamples; "
            "provide more varied ratings or disable bootstrapping."
        )
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(np.asarray(estimates), [tail, 1.0 - tail])
    return (float(low), float(high)), len(estimates)


def compute_interrater_reliability(
    codings: Sequence[Sequence[Any]],
    *,
    method: str = "krippendorff_alpha",
    n_bootstrap: int = 0,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> InterraterReliabilityResult:
    """Compute nominal inter-rater reliability from an item-by-rater matrix.

    ``None`` and ``NaN`` ratings are missing. Cohen's and Fleiss' kappa use
    complete items; Krippendorff's alpha uses every item with at least two
    observed ratings.
    """
    matrix = _coerce_matrix(codings)
    resolved_method = method.strip().lower()
    if n_bootstrap < 0 or (0 < n_bootstrap < 10):
        raise ValueError("n_bootstrap must be 0 or at least 10.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0, 1).")

    coefficient, n_items_used, n_observations, categories = _coefficient(
        matrix,
        method=resolved_method,
    )
    confidence_interval: tuple[float, float] | None = None
    bootstrap_samples = 0
    if n_bootstrap:
        confidence_interval, bootstrap_samples = _bootstrap_interval(
            matrix,
            method=resolved_method,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )

    n_raters = len(matrix[0])
    missing_ratings = sum(_is_missing(value) for row in matrix for value in row)
    return InterraterReliabilityResult(
        method=resolved_method,
        coefficient=coefficient,
        n_items=len(matrix),
        n_items_used=n_items_used,
        n_raters=n_raters,
        n_observations=n_observations,
        categories=categories,
        missing_ratings=missing_ratings,
        confidence_interval=confidence_interval,
        bootstrap_samples=bootstrap_samples,
        config={
            "measurement_level": "nominal",
            "missing_values": ["None", "NaN"],
            "item_policy": (
                "pairable" if resolved_method == "krippendorff_alpha" else "complete_case"
            ),
            "n_bootstrap_requested": int(n_bootstrap),
            "confidence_level": float(confidence_level),
            "seed": int(seed),
        },
    )
