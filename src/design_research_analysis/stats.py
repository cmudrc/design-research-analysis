"""Statistical analysis utilities for unified design-research workflows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from .table import coerce_unified_table

_STATS_IMPORT_ERROR = (
    "This statistical method requires optional dependencies. "
    "Install with `pip install design-research-analysis[stats]`."
)


@dataclass(slots=True)
class GroupComparisonResult:
    """Result container for group-comparison analyses."""

    method: str
    statistic: float
    p_value: float
    effect_size: float
    group_means: dict[str, float]
    group_sizes: dict[str, int]
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "method": self.method,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "effect_size": float(self.effect_size),
            "group_means": {k: float(v) for k, v in self.group_means.items()},
            "group_sizes": {k: int(v) for k, v in self.group_sizes.items()},
            "config": dict(self.config),
        }


@dataclass(slots=True)
class RegressionResult:
    """Result container for ordinary least squares regression."""

    coefficients: dict[str, float]
    intercept: float
    r2: float
    mse: float
    n_samples: int
    n_features: int
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "coefficients": {k: float(v) for k, v in self.coefficients.items()},
            "intercept": float(self.intercept),
            "r2": float(self.r2),
            "mse": float(self.mse),
            "n_samples": int(self.n_samples),
            "n_features": int(self.n_features),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class MixedEffectsResult:
    """Result container for mixed-effects model fitting."""

    success: bool
    backend: str
    formula: str
    group_column: str
    params: dict[str, float]
    aic: float | None
    bic: float | None
    log_likelihood: float | None
    message: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "success": bool(self.success),
            "backend": self.backend,
            "formula": self.formula,
            "group_column": self.group_column,
            "params": {k: float(v) for k, v in self.params.items()},
            "aic": None if self.aic is None else float(self.aic),
            "bic": None if self.bic is None else float(self.bic),
            "log_likelihood": None if self.log_likelihood is None else float(self.log_likelihood),
            "message": self.message,
            "config": dict(self.config),
        }


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _cohen_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    n_a = sample_a.size
    n_b = sample_b.size
    if n_a < 2 or n_b < 2:
        return 0.0

    var_a = float(np.var(sample_a, ddof=1))
    var_b = float(np.var(sample_b, ddof=1))
    pooled = (((n_a - 1) * var_a) + ((n_b - 1) * var_b)) / float(n_a + n_b - 2)
    if pooled <= 0.0:
        return 0.0
    return float((np.mean(sample_a) - np.mean(sample_b)) / np.sqrt(pooled))


def _eta_squared(groups: list[np.ndarray]) -> float:
    all_values = np.concatenate(groups)
    grand_mean = float(np.mean(all_values))
    ss_between = 0.0
    ss_total = 0.0
    for group in groups:
        group_mean = float(np.mean(group))
        ss_between += float(group.size) * (group_mean - grand_mean) ** 2
        ss_total += float(np.sum((group - grand_mean) ** 2))
    if ss_total == 0.0:
        return 0.0
    return float(ss_between / ss_total)


def compare_groups(
    values: Sequence[float] | None = None,
    groups: Sequence[Any] | None = None,
    *,
    data: Sequence[Mapping[str, Any]] | None = None,
    value_column: str = "value",
    group_column: str = "group",
    method: str = "auto",
) -> GroupComparisonResult:
    """Compare outcomes across groups using t-test or ANOVA.

    Args:
        values: Numeric values (row-aligned with ``groups``).
        groups: Group labels.
        data: Optional unified-table rows as an alternative to values/groups.
        value_column: Value column when ``data`` is provided.
        group_column: Group column when ``data`` is provided.
        method: ``auto``, ``ttest``, ``anova``, or ``kruskal``.
    """
    if data is not None:
        rows = coerce_unified_table(data)
        resolved_values: list[float] = []
        resolved_groups: list[str] = []
        for index, row in enumerate(rows):
            raw_value = row.get(value_column)
            raw_group = row.get(group_column)
            if _is_blank(raw_value) or _is_blank(raw_group):
                raise ValueError(f"Row {index} is missing '{value_column}' or '{group_column}'.")
            resolved_values.append(float(cast(float | int | str, raw_value)))
            resolved_groups.append(str(raw_group))
        values = resolved_values
        groups = resolved_groups

    if values is None or groups is None:
        raise ValueError("Provide either values/groups or table data with value/group columns.")
    if len(values) != len(groups):
        raise ValueError("values and groups must have the same length.")
    if len(values) == 0:
        raise ValueError("values must not be empty.")

    grouped_values: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        grouped_values[str(group)].append(float(value))

    if len(grouped_values) < 2:
        raise ValueError("At least two groups are required for comparison.")

    arrays = {group: np.asarray(vals, dtype=float) for group, vals in grouped_values.items()}
    group_means = {group: float(np.mean(arr)) for group, arr in arrays.items()}
    group_sizes = {group: int(arr.size) for group, arr in arrays.items()}

    try:
        from scipy import stats
    except ImportError as exc:
        raise ImportError(_STATS_IMPORT_ERROR) from exc

    ordered_groups = sorted(arrays)
    samples = [arrays[group] for group in ordered_groups]
    resolved_method = method.lower()

    if resolved_method == "auto":
        resolved_method = "ttest" if len(samples) == 2 else "anova"

    if resolved_method == "ttest":
        if len(samples) != 2:
            raise ValueError("ttest requires exactly two groups.")
        statistic, p_value = stats.ttest_ind(samples[0], samples[1], equal_var=False)
        effect_size = _cohen_d(samples[0], samples[1])
    elif resolved_method == "anova":
        statistic, p_value = stats.f_oneway(*samples)
        effect_size = _eta_squared(samples)
    elif resolved_method == "kruskal":
        statistic, p_value = stats.kruskal(*samples)
        effect_size = _eta_squared(samples)
    else:
        raise ValueError("Unsupported method. Valid options: auto, ttest, anova, kruskal.")

    return GroupComparisonResult(
        method=resolved_method,
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=float(effect_size),
        group_means=group_means,
        group_sizes=group_sizes,
        config={
            "n_groups": len(samples),
            "n_samples": len(values),
            "value_column": value_column,
            "group_column": group_column,
        },
    )


def fit_regression(
    X: Sequence[Sequence[float]] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    add_intercept: bool = True,
) -> RegressionResult:
    """Fit an ordinary least squares regression model with ``numpy``."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    if y_arr.ndim != 1:
        raise ValueError("y must be a 1D vector.")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if X_arr.shape[0] == 0:
        raise ValueError("X and y must not be empty.")

    design = X_arr
    intercept = 0.0
    if add_intercept:
        ones = np.ones((X_arr.shape[0], 1), dtype=float)
        design = np.hstack([ones, X_arr])

    coefficients, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    predictions = design @ coefficients
    residuals = y_arr - predictions

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 0.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)
    mse = float(np.mean(residuals**2))

    if add_intercept:
        intercept = float(coefficients[0])
        weight_vector = coefficients[1:]
    else:
        weight_vector = coefficients

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
    if len(feature_names) != X_arr.shape[1]:
        raise ValueError("feature_names length must match number of features in X.")

    weights = {
        str(name): float(value) for name, value in zip(feature_names, weight_vector, strict=True)
    }
    return RegressionResult(
        coefficients=weights,
        intercept=intercept,
        r2=float(r2),
        mse=float(mse),
        n_samples=int(X_arr.shape[0]),
        n_features=int(X_arr.shape[1]),
        config={"add_intercept": bool(add_intercept)},
    )


def fit_mixed_effects(
    data: Sequence[Mapping[str, Any]],
    *,
    formula: str,
    group_column: str,
    backend: str = "statsmodels",
    reml: bool = True,
    max_iter: int = 200,
) -> MixedEffectsResult:
    """Fit a mixed-effects model using ``statsmodels``.

    Args:
        data: Unified-table rows.
        formula: Patsy-style model formula (e.g., ``outcome ~ condition``).
        group_column: Random-effect grouping column.
        backend: Backend name. Currently only ``statsmodels`` is supported.
        reml: Whether to use REML estimation.
        max_iter: Maximum optimizer iterations.
    """
    if backend != "statsmodels":
        raise ValueError("Unsupported backend. Valid options: statsmodels.")

    rows = coerce_unified_table(data)
    if not rows:
        raise ValueError("Mixed-effects fitting requires at least one row.")

    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError(_STATS_IMPORT_ERROR) from exc

    frame = pd.DataFrame(rows)
    if group_column not in frame.columns:
        raise ValueError(f"group_column '{group_column}' was not found in the table.")

    model = smf.mixedlm(formula=formula, data=frame, groups=frame[group_column])
    try:
        fitted = model.fit(reml=reml, maxiter=max_iter, method="lbfgs")
    except Exception as exc:  # pragma: no cover - backend-dependent failure modes
        return MixedEffectsResult(
            success=False,
            backend="statsmodels",
            formula=formula,
            group_column=group_column,
            params={},
            aic=None,
            bic=None,
            log_likelihood=None,
            message=str(exc),
            config={"reml": bool(reml), "max_iter": int(max_iter)},
        )

    params = {str(name): float(value) for name, value in fitted.params.items()}
    return MixedEffectsResult(
        success=True,
        backend="statsmodels",
        formula=formula,
        group_column=group_column,
        params=params,
        aic=float(fitted.aic) if fitted.aic is not None else None,
        bic=float(fitted.bic) if fitted.bic is not None else None,
        log_likelihood=float(fitted.llf) if fitted.llf is not None else None,
        message="",
        config={"reml": bool(reml), "max_iter": int(max_iter)},
    )


__all__ = [
    "GroupComparisonResult",
    "MixedEffectsResult",
    "RegressionResult",
    "compare_groups",
    "fit_mixed_effects",
    "fit_regression",
]
