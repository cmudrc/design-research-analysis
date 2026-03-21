"""Statistical analysis utilities for unified design-research workflows."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ._comparison import ComparableResultMixin
from .table import UnifiedTableConfig, coerce_unified_table

if TYPE_CHECKING:
    import pandas as pd

_STATS_IMPORT_ERROR = (
    "This statistical method requires optional dependencies. "
    "Install with `pip install design-research-analysis[stats]`."
)
_POWER_IMPORT_ERROR = (
    "Power analysis requires optional dependencies. "
    "Install with `pip install design-research-analysis[stats]`."
)
_SUPPORTED_TESTS = {"one_sample_t", "paired_t", "two_sample_t"}
_SUPPORTED_ALTERNATIVES = {"two-sided", "larger", "smaller"}
_SUPPORTED_PERMUTATION_ALTERNATIVES = {"two-sided", "greater", "less"}
_ANALYSIS_TABLE_CONFIG = UnifiedTableConfig(
    required_columns=(),
    recommended_columns=(),
    optional_columns=(),
    parse_timestamps=False,
    sort_by_timestamp=False,
)
_DEFAULT_EXACT_PERMUTATION_THRESHOLD = 250_000
_DEFAULT_SAMPLED_PERMUTATIONS = 20_000


@dataclass(slots=True)
class GroupComparisonResult(ComparableResultMixin):
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

    def _comparison_metric(self) -> str:
        return "group_summary"

    def _comparison_vectors(
        self,
        other: GroupComparisonResult,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        labels = sorted(set(self.group_means) | set(other.group_means))
        left_means = np.asarray(
            [self.group_means.get(label, 0.0) for label in labels],
            dtype=float,
        )
        right_means = np.asarray(
            [other.group_means.get(label, 0.0) for label in labels],
            dtype=float,
        )
        left_sizes = np.asarray(
            [self.group_sizes.get(label, 0) for label in labels],
            dtype=float,
        )
        right_sizes = np.asarray(
            [other.group_sizes.get(label, 0) for label in labels],
            dtype=float,
        )
        left_vector = np.concatenate(
            [left_means, left_sizes, np.asarray([self.statistic, self.effect_size], dtype=float)]
        )
        right_vector = np.concatenate(
            [
                right_means,
                right_sizes,
                np.asarray([other.statistic, other.effect_size], dtype=float),
            ]
        )
        return (
            left_vector,
            right_vector,
            {
                "group_labels": labels,
                "methods": [self.method, other.method],
            },
        )


@dataclass(slots=True)
class RegressionResult(ComparableResultMixin):
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

    def _comparison_metric(self) -> str:
        return "regression_profile"

    def _comparison_vectors(
        self,
        other: RegressionResult,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        names = sorted(set(self.coefficients) | set(other.coefficients))
        left_coeffs = np.asarray(
            [self.coefficients.get(name, 0.0) for name in names],
            dtype=float,
        )
        right_coeffs = np.asarray(
            [other.coefficients.get(name, 0.0) for name in names],
            dtype=float,
        )
        left_vector = np.concatenate(
            [
                np.asarray([self.intercept], dtype=float),
                left_coeffs,
                np.asarray([self.r2, self.mse]),
            ]
        )
        right_vector = np.concatenate(
            [
                np.asarray([other.intercept], dtype=float),
                right_coeffs,
                np.asarray([other.r2, other.mse]),
            ]
        )
        return left_vector, right_vector, {"feature_names": names}


@dataclass(slots=True)
class MixedEffectsResult(ComparableResultMixin):
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

    def _comparison_metric(self) -> str:
        return "mixed_effects_profile"

    def _comparison_vectors(
        self,
        other: MixedEffectsResult,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        names = sorted(set(self.params) | set(other.params))
        left_params = np.asarray([self.params.get(name, 0.0) for name in names], dtype=float)
        right_params = np.asarray([other.params.get(name, 0.0) for name in names], dtype=float)
        left_tail = np.asarray(
            [
                float(self.success),
                0.0 if self.aic is None else self.aic,
                0.0 if self.bic is None else self.bic,
                0.0 if self.log_likelihood is None else self.log_likelihood,
            ],
            dtype=float,
        )
        right_tail = np.asarray(
            [
                float(other.success),
                0.0 if other.aic is None else other.aic,
                0.0 if other.bic is None else other.bic,
                0.0 if other.log_likelihood is None else other.log_likelihood,
            ],
            dtype=float,
        )
        return (
            np.concatenate([left_params, left_tail]),
            np.concatenate([right_params, right_tail]),
            {
                "formulae": [self.formula, other.formula],
                "group_columns": [self.group_column, other.group_column],
                "parameter_names": names,
            },
        )


@dataclass(frozen=True, slots=True)
class ConditionPairComparison:
    """Structured result for one pairwise condition comparison."""

    metric: str
    left_condition: str
    right_condition: str
    mean_left: float
    mean_right: float
    n_left: int
    n_right: int
    mean_difference: float
    effect_size: float
    p_value: float
    alternative: str
    test_method: str
    permutations_evaluated: int
    total_permutations: int
    higher_condition: str | None
    significant: bool

    @property
    def pair_label(self) -> str:
        """Return a stable display label for this pair."""
        return f"{self.left_condition} vs {self.right_condition}"

    @property
    def test_name(self) -> str:
        """Return the test label expected by reporting helpers."""
        if self.test_method == "exact":
            return "exact_permutation_test"
        return "sampled_permutation_test"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "metric": self.metric,
            "left_condition": self.left_condition,
            "right_condition": self.right_condition,
            "pair_label": self.pair_label,
            "mean_left": float(self.mean_left),
            "mean_right": float(self.mean_right),
            "n_left": int(self.n_left),
            "n_right": int(self.n_right),
            "mean_difference": float(self.mean_difference),
            "effect_size": float(self.effect_size),
            "p_value": float(self.p_value),
            "alternative": self.alternative,
            "test_method": self.test_method,
            "test_name": self.test_name,
            "permutations_evaluated": int(self.permutations_evaluated),
            "total_permutations": int(self.total_permutations),
            "higher_condition": self.higher_condition,
            "significant": bool(self.significant),
        }


@dataclass(frozen=True, slots=True)
class ConditionComparisonReport:
    """Structured report for a set of pairwise condition comparisons."""

    metric: str
    condition_column: str
    metric_column: str
    alternative: str
    alpha: float
    comparisons: tuple[ConditionPairComparison, ...]
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "metric": self.metric,
            "condition_column": self.condition_column,
            "metric_column": self.metric_column,
            "alternative": self.alternative,
            "alpha": float(self.alpha),
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "config": dict(self.config),
        }

    def to_significance_rows(self) -> list[dict[str, Any]]:
        """Render comparison rows compatible with experiments reporting helpers."""
        rows: list[dict[str, Any]] = []
        for comparison in self.comparisons:
            rows.append(
                {
                    "test": comparison.test_name,
                    "outcome": f"{self.metric} ({comparison.pair_label})",
                    "p_value": float(comparison.p_value),
                    "effect_size": float(comparison.effect_size),
                    "mean_difference": float(comparison.mean_difference),
                    "higher_condition": comparison.higher_condition,
                    "alternative": comparison.alternative,
                    "test_method": comparison.test_method,
                    "permutations_evaluated": int(comparison.permutations_evaluated),
                    "total_permutations": int(comparison.total_permutations),
                    "significant": bool(comparison.significant),
                }
            )
        return rows

    def render_brief(self) -> str:
        """Render a concise markdown brief for the comparison set."""
        lines = ["## Condition Comparison Brief"]
        if not self.comparisons:
            lines.append("- No condition comparisons were produced.")
            return "\n".join(lines)

        for comparison in self.comparisons:
            significant = "yes" if comparison.significant else "no"
            lines.append(
                "- "
                f"`{comparison.pair_label}` on `{self.metric}`: "
                f"{comparison.left_condition} mean={comparison.mean_left:.4g}, "
                f"{comparison.right_condition} mean={comparison.mean_right:.4g}, "
                f"diff={comparison.mean_difference:+.4g}, "
                f"d={comparison.effect_size:+.4g}, "
                f"p={comparison.p_value:.4g} "
                f"({comparison.test_method}, {comparison.alternative}, significant={significant})."
            )
        return "\n".join(lines)


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


def _as_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return arr


def _calc_stat(x: np.ndarray, y: np.ndarray | None, stat: str) -> float:
    if stat == "mean":
        return float(np.mean(x))
    if stat == "median":
        return float(np.median(x))
    if y is None:
        raise ValueError(f"stat '{stat}' requires y values.")
    if stat == "diff_means":
        return float(np.mean(x) - np.mean(y))
    if stat == "diff_medians":
        return float(np.median(x) - np.median(y))
    raise ValueError("stat must be one of: mean, median, diff_means, diff_medians")


def _render_ci_text(estimate: float, ci_low: float, ci_high: float, ci: float, stat: str) -> str:
    return (
        f"The bootstrap {stat} estimate is {estimate:.4g}. "
        f"A {ci * 100:.1f}% interval spans [{ci_low:.4g}, {ci_high:.4g}], "
        "which describes uncertainty under the resampling assumptions."
    )


def _render_permutation_text(p_value: float, stat_name: str, alternative: str) -> str:
    return (
        f"The permutation test for {stat_name} produced p={p_value:.4g} ({alternative}). "
        "Interpret this as evidence against the null random-label model, not as proof of causality."
    )


def _render_np_test_text(
    test_name: str,
    p_value: float,
    alpha: float,
    effect_size: float | None,
) -> str:
    decision = "below" if p_value < alpha else "above"
    effect_txt = ""
    if effect_size is not None:
        effect_txt = f" Effect size estimate is {effect_size:.4g}."
    return (
        f"{test_name} returned p={p_value:.4g}, which is {decision} alpha={alpha:.3g}."
        f"{effect_txt} Use this with distribution checks and study context."
    )


def _coerce_analysis_rows(data: Any, *, table_name: str) -> list[dict[str, Any]]:
    try:
        return coerce_unified_table(data, config=_ANALYSIS_TABLE_CONFIG)
    except ValueError as exc:
        raise ValueError(f"Failed to coerce {table_name}: {exc}") from exc


def _unique_row_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_column: str,
    table_name: str,
) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        raw_key = row.get(key_column)
        if _is_blank(raw_key):
            raise ValueError(f"{table_name} row {index} is missing '{key_column}'.")
        key = str(raw_key)
        if key in resolved:
            raise ValueError(f"{table_name} contains duplicate '{key_column}' value {key!r}.")
        resolved[key] = dict(row)
    return resolved


def _collect_rows_by_run_id(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_id_column: str,
    table_name: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        raw_run_id = row.get(run_id_column)
        if _is_blank(raw_run_id):
            raise ValueError(f"{table_name} row {index} is missing '{run_id_column}'.")
        grouped[str(raw_run_id)].append(dict(row))
    return dict(grouped)


def _resolve_metric_label(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_name: str | None,
    metric_label_column: str,
    metric_column: str,
) -> str:
    if metric_name is not None and metric_name.strip():
        return metric_name

    labels = {
        str(value)
        for row in rows
        if metric_label_column in row and not _is_blank(value := row.get(metric_label_column))
    }
    if len(labels) == 1:
        return next(iter(labels))
    if len(labels) > 1:
        raise ValueError(
            "Joined table contains multiple metric labels. "
            "Provide metric_name explicitly or pre-filter the table."
        )
    return metric_column


def _resolve_condition_pairs(
    grouped_values: Mapping[str, Sequence[float]],
    *,
    condition_pairs: Sequence[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    known_conditions = sorted(grouped_values)
    if condition_pairs is None:
        return list(combinations(known_conditions, 2))

    resolved_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, pair in enumerate(condition_pairs):
        if len(pair) != 2:
            raise ValueError(f"condition_pairs[{index}] must contain exactly two condition labels.")
        left, right = str(pair[0]), str(pair[1])
        if left == right:
            raise ValueError(f"condition_pairs[{index}] compares {left!r} against itself.")
        if left not in grouped_values or right not in grouped_values:
            raise ValueError(
                f"condition_pairs[{index}] references unknown conditions: {(left, right)!r}."
            )
        normalized_pair = (left, right)
        if normalized_pair in seen_pairs:
            raise ValueError(f"condition_pairs[{index}] duplicates pair {normalized_pair!r}.")
        seen_pairs.add(normalized_pair)
        resolved_pairs.append(normalized_pair)
    return resolved_pairs


def _exact_permutation_summary(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    *,
    alternative: str,
) -> dict[str, Any]:
    pooled = np.concatenate([x_arr, y_arr])
    observed = float(np.mean(x_arr) - np.mean(y_arr))
    n_x = int(x_arr.size)
    n_total = int(pooled.size)
    total_permutations = math.comb(n_total, n_x)
    pooled_sum = float(np.sum(pooled))
    exceedances = 0

    for left_indices in combinations(range(n_total), n_x):
        left_sum = float(sum(float(pooled[index]) for index in left_indices))
        right_sum = pooled_sum - left_sum
        stat = (left_sum / n_x) - (right_sum / (n_total - n_x))
        if alternative == "two-sided":
            is_extreme = abs(stat) >= abs(observed) - 1e-12
        elif alternative == "greater":
            is_extreme = stat >= observed - 1e-12
        else:
            is_extreme = stat <= observed + 1e-12
        if is_extreme:
            exceedances += 1

    return {
        "p_value": float(exceedances / total_permutations),
        "test_method": "exact",
        "permutations_evaluated": int(total_permutations),
        "total_permutations": int(total_permutations),
    }


def _pairwise_permutation_summary(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    *,
    alternative: str,
    exact_threshold: int,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    total_permutations = math.comb(int(x_arr.size + y_arr.size), int(x_arr.size))
    if total_permutations <= exact_threshold:
        return _exact_permutation_summary(x_arr, y_arr, alternative=alternative)

    sampled = permutation_test(
        x_arr,
        y_arr,
        n_permutations=n_permutations,
        alternative=alternative,
        seed=seed,
    )
    return {
        "p_value": float(sampled["p_value"]),
        "test_method": "sampled",
        "permutations_evaluated": int(n_permutations),
        "total_permutations": int(total_permutations),
    }


def compare_groups(
    values: Sequence[float] | None = None,
    groups: Sequence[Any] | None = None,
    *,
    data: Sequence[Mapping[str, Any]] | None = None,
    value_column: str = "value",
    group_column: str = "group",
    method: str = "auto",
) -> GroupComparisonResult:
    """Compare outcomes across groups using t-test, ANOVA, or Kruskal-Wallis."""
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
    """Fit a mixed-effects model using ``statsmodels``."""
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


def bootstrap_ci(
    x: Any,
    *,
    stat: str = "mean",
    y: Any | None = None,
    n_resamples: int = 10000,
    ci: float = 0.95,
    method: str = "percentile",
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate bootstrap confidence intervals for one- and two-sample statistics."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1).")

    x_arr = _as_array(x, "x")
    y_arr = _as_array(y, "y") if y is not None else None

    if method == "bca":
        try:
            from scipy import stats as sp_stats
        except ImportError as exc:
            raise ImportError(_STATS_IMPORT_ERROR) from exc

        alpha = 1.0 - ci
        if y_arr is None:
            stat_fun: Any
            if stat == "mean":
                stat_fun = np.mean
            elif stat == "median":
                stat_fun = np.median
            else:
                raise ValueError(f"stat '{stat}' requires y values.")
            bres = sp_stats.bootstrap(
                (x_arr,),
                stat_fun,
                confidence_level=ci,
                n_resamples=n_resamples,
                method="BCa",
                random_state=seed,
            )
            ci_low = float(bres.confidence_interval.low)
            ci_high = float(bres.confidence_interval.high)
            estimate = _calc_stat(x_arr, y_arr, stat)
            rng = np.random.default_rng(seed)
            samples = np.empty(n_resamples, dtype=float)
            n_x = x_arr.size
            for idx in range(n_resamples):
                boot_x = x_arr[rng.integers(0, n_x, size=n_x)]
                samples[idx] = _calc_stat(boot_x, None, stat)
        else:
            if stat not in {"diff_means", "diff_medians"}:
                raise ValueError(f"stat '{stat}' is not valid for two-sample BCa.")
            rng = np.random.default_rng(seed)
            samples = np.empty(n_resamples, dtype=float)
            n_x = x_arr.size
            n_y = y_arr.size
            for idx in range(n_resamples):
                boot_x = x_arr[rng.integers(0, n_x, size=n_x)]
                boot_y = y_arr[rng.integers(0, n_y, size=n_y)]
                samples[idx] = _calc_stat(boot_x, boot_y, stat)
            ci_low = float(np.quantile(samples, alpha / 2.0))
            ci_high = float(np.quantile(samples, 1.0 - alpha / 2.0))
            estimate = _calc_stat(x_arr, y_arr, stat)

        distribution_summary = {
            "n": int(samples.size),
            "std": float(np.std(samples, ddof=1)),
            "skew": float(
                ((samples - samples.mean()) ** 3).mean() / (np.std(samples) ** 3 + 1e-12)
            ),
        }
        return {
            "estimate": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "distribution_summary": distribution_summary,
            "method_used": method,
            "interpretation": _render_ci_text(
                estimate=estimate,
                ci_low=ci_low,
                ci_high=ci_high,
                ci=ci,
                stat=stat,
            ),
        }

    if method != "percentile":
        raise ValueError("method must be one of: percentile, bca")

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    n_x = x_arr.size
    n_y = y_arr.size if y_arr is not None else 0
    for idx in range(n_resamples):
        boot_x = x_arr[rng.integers(0, n_x, size=n_x)]
        boot_y_opt = y_arr[rng.integers(0, n_y, size=n_y)] if y_arr is not None else None
        samples[idx] = _calc_stat(boot_x, boot_y_opt, stat)

    alpha = 1.0 - ci
    ci_low = float(np.quantile(samples, alpha / 2.0))
    ci_high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    estimate = _calc_stat(x_arr, y_arr, stat)

    distribution_summary = {
        "n": int(samples.size),
        "std": float(np.std(samples, ddof=1)),
        "skew": float(((samples - samples.mean()) ** 3).mean() / (np.std(samples) ** 3 + 1e-12)),
    }
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "distribution_summary": distribution_summary,
        "method_used": method,
        "interpretation": _render_ci_text(
            estimate=estimate,
            ci_low=ci_low,
            ci_high=ci_high,
            ci=ci,
            stat=stat,
        ),
    }


def permutation_test(
    x: Any,
    y: Any,
    *,
    stat: str = "diff_means",
    n_permutations: int = 20000,
    alternative: str = "two-sided",
    seed: int = 0,
) -> dict[str, Any]:
    """Run a two-sample permutation test."""
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")
    if alternative not in _SUPPORTED_PERMUTATION_ALTERNATIVES:
        raise ValueError("alternative must be one of: two-sided, greater, less")

    x_arr = _as_array(x, "x")
    y_arr = _as_array(y, "y")
    observed = _calc_stat(x_arr, y_arr, stat)

    rng = np.random.default_rng(seed)
    pooled = np.concatenate([x_arr, y_arr])
    n_x = x_arr.size
    perm_stats = np.empty(n_permutations, dtype=float)

    for idx in range(n_permutations):
        perm = rng.permutation(pooled)
        perm_stats[idx] = _calc_stat(perm[:n_x], perm[n_x:], stat)

    if alternative == "two-sided":
        p_value = float((np.abs(perm_stats) >= abs(observed)).mean())
    elif alternative == "greater":
        p_value = float((perm_stats >= observed).mean())
    else:
        p_value = float((perm_stats <= observed).mean())

    return {
        "p_value": p_value,
        "observed_stat": float(observed),
        "stat": stat,
        "alternative": alternative,
        "interpretation": _render_permutation_text(
            p_value=p_value,
            stat_name=stat,
            alternative=alternative,
        ),
    }


def build_condition_metric_table(
    runs: Any,
    *,
    metric: str,
    condition_column: str = "condition",
    evaluations: Any | None = None,
    conditions: Any | None = None,
    run_id_column: str = "run_id",
    condition_id_column: str = "condition_id",
    evaluation_metric_column: str = "metric_name",
    evaluation_value_column: str = "metric_value",
) -> list[dict[str, Any]]:
    """Build a normalized run-level condition/metric table from experiment exports."""
    run_rows = _coerce_analysis_rows(runs, table_name="runs table")
    if not run_rows:
        raise ValueError("runs table must contain at least one row.")

    condition_rows = (
        _coerce_analysis_rows(conditions, table_name="conditions table")
        if conditions is not None
        else []
    )
    evaluation_rows = (
        _coerce_analysis_rows(evaluations, table_name="evaluations table")
        if evaluations is not None
        else []
    )

    condition_lookup = (
        _unique_row_map(
            condition_rows,
            key_column=condition_id_column,
            table_name="conditions table",
        )
        if condition_rows
        else {}
    )
    evaluation_lookup = (
        _collect_rows_by_run_id(
            evaluation_rows,
            run_id_column=run_id_column,
            table_name="evaluations table",
        )
        if evaluation_rows
        else {}
    )

    metric_in_runs = any(metric in row for row in run_rows)
    condition_in_runs = any(condition_column in row for row in run_rows)
    normalized: list[dict[str, Any]] = []

    for index, row in enumerate(run_rows):
        raw_run_id = row.get(run_id_column)
        if _is_blank(raw_run_id):
            raise ValueError(f"runs table row {index} is missing '{run_id_column}'.")
        run_id = str(raw_run_id)

        raw_condition_id = row.get(condition_id_column)
        condition_id = "" if _is_blank(raw_condition_id) else str(raw_condition_id)

        if condition_in_runs:
            raw_condition = row.get(condition_column)
            if _is_blank(raw_condition):
                raise ValueError(
                    "runs table row "
                    f"{index} is missing direct condition column {condition_column!r}."
                )
            condition_value = str(raw_condition)
            condition_source = "runs"
        else:
            if not condition_lookup:
                raise ValueError(
                    f"Condition column {condition_column!r} was not found in runs rows and no "
                    "conditions table was provided."
                )
            if not condition_id:
                raise ValueError(
                    f"runs table row {index} is missing '{condition_id_column}' "
                    "for conditions join."
                )
            condition_row = condition_lookup.get(condition_id)
            if condition_row is None:
                raise ValueError(
                    f"runs table row {index} references unknown condition_id {condition_id!r}."
                )
            raw_condition = condition_row.get(condition_column)
            if _is_blank(raw_condition):
                raise ValueError(
                    f"conditions row for condition_id {condition_id!r} is missing "
                    f"{condition_column!r}."
                )
            condition_value = str(raw_condition)
            condition_source = "conditions"

        if metric_in_runs:
            raw_value = row.get(metric)
            if _is_blank(raw_value):
                raise ValueError(f"runs table row {index} is missing metric column {metric!r}.")
            metric_value = float(cast(float | int | str, raw_value))
            metric_source = "runs"
        else:
            if not evaluation_lookup:
                raise ValueError(
                    f"Metric {metric!r} was not found in runs rows and no evaluations table "
                    "was provided."
                )
            matching_rows = []
            for evaluation_row in evaluation_lookup.get(run_id, []):
                aggregation_level = evaluation_row.get("aggregation_level")
                if not _is_blank(aggregation_level) and str(aggregation_level) != "run":
                    continue
                if str(evaluation_row.get(evaluation_metric_column, "")) == metric:
                    matching_rows.append(evaluation_row)

            if not matching_rows:
                raise ValueError(
                    f"No evaluation metric {metric!r} was found for run_id {run_id!r}."
                )
            if len(matching_rows) > 1:
                raise ValueError(
                    f"Multiple evaluation rows matched metric {metric!r} for run_id {run_id!r}."
                )
            raw_value = matching_rows[0].get(evaluation_value_column)
            if _is_blank(raw_value):
                raise ValueError(
                    f"Evaluation metric {metric!r} for run_id {run_id!r} is missing "
                    f"{evaluation_value_column!r}."
                )
            metric_value = float(cast(float | int | str, raw_value))
            metric_source = "evaluations"

        normalized.append(
            {
                "run_id": run_id,
                "condition_id": condition_id,
                "condition": condition_value,
                "metric": metric,
                "value": metric_value,
                "condition_source": condition_source,
                "metric_source": metric_source,
            }
        )

    return normalized


def compare_condition_pairs(
    data: Any,
    *,
    condition_column: str = "condition",
    metric_column: str = "value",
    metric_name: str | None = None,
    condition_pairs: Sequence[tuple[str, str]] | None = None,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    exact_threshold: int = _DEFAULT_EXACT_PERMUTATION_THRESHOLD,
    n_permutations: int = _DEFAULT_SAMPLED_PERMUTATIONS,
    seed: int = 0,
) -> ConditionComparisonReport:
    """Compare all or selected condition pairs on one numeric metric."""
    if alternative not in _SUPPORTED_PERMUTATION_ALTERNATIVES:
        raise ValueError("alternative must be one of: two-sided, greater, less")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if exact_threshold <= 0:
        raise ValueError("exact_threshold must be positive.")
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")

    rows = _coerce_analysis_rows(data, table_name="comparison table")
    if not rows:
        raise ValueError("comparison table must contain at least one row.")

    grouped_values: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        raw_condition = row.get(condition_column)
        raw_value = row.get(metric_column)
        if _is_blank(raw_condition):
            raise ValueError(f"comparison table row {index} is missing '{condition_column}'.")
        if _is_blank(raw_value):
            raise ValueError(f"comparison table row {index} is missing '{metric_column}'.")
        grouped_values[str(raw_condition)].append(float(cast(float | int | str, raw_value)))

    if len(grouped_values) < 2:
        raise ValueError("At least two conditions are required for pairwise comparison.")

    resolved_metric = _resolve_metric_label(
        rows,
        metric_name=metric_name,
        metric_label_column="metric",
        metric_column=metric_column,
    )
    resolved_pairs = _resolve_condition_pairs(
        grouped_values,
        condition_pairs=condition_pairs,
    )
    if not resolved_pairs:
        raise ValueError("No condition pairs were requested.")

    comparisons: list[ConditionPairComparison] = []
    for pair_index, (left_condition, right_condition) in enumerate(resolved_pairs):
        left_arr = np.asarray(grouped_values[left_condition], dtype=float)
        right_arr = np.asarray(grouped_values[right_condition], dtype=float)
        mean_left = float(np.mean(left_arr))
        mean_right = float(np.mean(right_arr))
        mean_difference = mean_left - mean_right
        effect_size = _cohen_d(left_arr, right_arr)
        permutation_summary = _pairwise_permutation_summary(
            left_arr,
            right_arr,
            alternative=alternative,
            exact_threshold=exact_threshold,
            n_permutations=n_permutations,
            seed=seed + pair_index,
        )
        if math.isclose(mean_difference, 0.0, abs_tol=1e-12):
            higher_condition = None
        elif mean_difference > 0:
            higher_condition = left_condition
        else:
            higher_condition = right_condition

        comparisons.append(
            ConditionPairComparison(
                metric=resolved_metric,
                left_condition=left_condition,
                right_condition=right_condition,
                mean_left=mean_left,
                mean_right=mean_right,
                n_left=int(left_arr.size),
                n_right=int(right_arr.size),
                mean_difference=mean_difference,
                effect_size=effect_size,
                p_value=float(permutation_summary["p_value"]),
                alternative=alternative,
                test_method=str(permutation_summary["test_method"]),
                permutations_evaluated=int(permutation_summary["permutations_evaluated"]),
                total_permutations=int(permutation_summary["total_permutations"]),
                higher_condition=higher_condition,
                significant=float(permutation_summary["p_value"]) < alpha,
            )
        )

    return ConditionComparisonReport(
        metric=resolved_metric,
        condition_column=condition_column,
        metric_column=metric_column,
        alternative=alternative,
        alpha=float(alpha),
        comparisons=tuple(comparisons),
        config={
            "n_conditions": len(grouped_values),
            "exact_threshold": int(exact_threshold),
            "n_permutations": int(n_permutations),
            "seed": int(seed),
        },
    )


def rank_tests_one_stop(
    x: Any,
    y: Any | None = None,
    groups: Sequence[Any] | None = None,
    *,
    paired: bool | None = None,
    kind: str | None = None,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Dispatch to a nonparametric rank test with consistent structured output."""
    try:
        from scipy import stats as sp_stats
    except ImportError as exc:
        raise ImportError(_STATS_IMPORT_ERROR) from exc

    x_arr = _as_array(x, "x")
    if kind is None:
        if groups is not None:
            kind = "friedman" if paired else "kruskal"
        elif y is not None:
            kind = "wilcoxon" if paired else "mannwhitney"
        else:
            raise ValueError("Provide y or groups to select a rank test.")

    effect_size: float | None = None
    result: dict[str, Any]
    guidance = [
        "Report exact sample sizes and handling of ties.",
        "Pair p-values with effect size and uncertainty context.",
        "Avoid causal claims from rank tests alone.",
    ]

    if kind == "mannwhitney":
        if y is None:
            raise ValueError("mannwhitney requires y.")
        y_arr = _as_array(y, "y")
        stat_val, p_value = sp_stats.mannwhitneyu(x_arr, y_arr, alternative=alternative)
        n1, n2 = x_arr.size, y_arr.size
        effect_size = 1.0 - 2.0 * float(stat_val) / float(n1 * n2)
        result = {"test": "mannwhitney", "statistic": float(stat_val), "p_value": float(p_value)}
    elif kind == "wilcoxon":
        if y is None:
            raise ValueError("wilcoxon requires y.")
        y_arr = _as_array(y, "y")
        stat_val, p_value = sp_stats.wilcoxon(x_arr, y_arr, alternative=alternative)
        n = x_arr.size
        max_w = n * (n + 1) / 2.0
        effect_size = 1.0 - 2.0 * float(stat_val) / max_w
        result = {"test": "wilcoxon", "statistic": float(stat_val), "p_value": float(p_value)}
    elif kind == "kruskal":
        if groups is None:
            raise ValueError("kruskal requires groups as list-like samples.")
        arrays = [_as_array(g, f"group_{idx}") for idx, g in enumerate(groups)]
        stat_val, p_value = sp_stats.kruskal(*arrays)
        n_total = int(sum(len(g) for g in arrays))
        k = len(arrays)
        effect_size = float((stat_val - k + 1) / (n_total - k)) if n_total > k else 0.0
        result = {"test": "kruskal", "statistic": float(stat_val), "p_value": float(p_value)}
    elif kind == "friedman":
        if groups is None:
            raise ValueError("friedman requires groups as repeated-measures samples.")
        arrays = [_as_array(g, f"group_{idx}") for idx, g in enumerate(groups)]
        stat_val, p_value = sp_stats.friedmanchisquare(*arrays)
        n = len(arrays[0])
        k = len(arrays)
        effect_size = float(stat_val / (n * (k - 1))) if n > 0 and k > 1 else 0.0
        result = {"test": "friedman", "statistic": float(stat_val), "p_value": float(p_value)}
    else:
        raise ValueError("kind must be one of: mannwhitney, wilcoxon, kruskal, friedman")

    result["effect_size"] = effect_size
    result["alpha"] = alpha
    result["interpretation"] = _render_np_test_text(
        test_name=result["test"],
        p_value=result["p_value"],
        alpha=alpha,
        effect_size=effect_size,
    )
    result["reporting_guidance"] = guidance
    return result


def _load_power_engines() -> tuple[Any, Any]:
    try:
        from statsmodels.stats.power import TTestIndPower, TTestPower
    except ImportError as exc:
        raise ImportError(_POWER_IMPORT_ERROR) from exc
    return TTestIndPower(), TTestPower()


def _validate_common_inputs(
    *,
    test: str,
    alpha: float,
    power: float | None = None,
    ratio: float = 1.0,
    alternative: str,
) -> None:
    if test not in _SUPPORTED_TESTS:
        valid = ", ".join(sorted(_SUPPORTED_TESTS))
        raise ValueError(f"test must be one of: {valid}")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if power is not None and not (0.0 < power < 1.0):
        raise ValueError("power must be in (0, 1).")
    if ratio <= 0:
        raise ValueError("ratio must be positive.")
    if alternative not in _SUPPORTED_ALTERNATIVES:
        valid = ", ".join(sorted(_SUPPORTED_ALTERNATIVES))
        raise ValueError(f"alternative must be one of: {valid}")


def _two_sample_allocation_from_n(total_n: int, ratio: float) -> tuple[int, int]:
    n1 = max(1, round(total_n / (1.0 + ratio)))
    if n1 >= total_n:
        n1 = total_n - 1
    n2 = total_n - n1
    if n2 <= 0:
        n2 = 1
        n1 = total_n - n2
    return n1, n2


def _compute_power(
    effect_size: float,
    *,
    n: int,
    test: str,
    alpha: float,
    ratio: float,
    alternative: str,
) -> float:
    ind_power, one_power = _load_power_engines()
    effect = abs(effect_size)
    if test == "two_sample_t":
        n1, n2 = _two_sample_allocation_from_n(n, ratio)
        return float(
            ind_power.power(
                effect_size=effect,
                nobs1=n1,
                alpha=alpha,
                ratio=(n2 / n1),
                alternative=alternative,
            )
        )
    return float(one_power.power(effect_size=effect, nobs=n, alpha=alpha, alternative=alternative))


def estimate_sample_size(
    effect_size: float,
    *,
    test: str,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Estimate total sample size for supported t-test families."""
    _validate_common_inputs(
        test=test,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative,
    )

    effect = abs(effect_size)
    if effect == 0:
        raise ValueError("effect_size must be non-zero.")

    ind_power, one_power = _load_power_engines()
    assumptions = ["Effect size is interpreted as Cohen's d."]

    if test == "two_sample_t":
        nobs1 = float(
            ind_power.solve_power(
                effect_size=effect,
                nobs1=None,
                alpha=alpha,
                power=power,
                ratio=ratio,
                alternative=alternative,
            )
        )
        n1 = max(1, math.ceil(nobs1))
        n2 = max(1, math.ceil(n1 * ratio))
        recommended_n = n1 + n2
        group_allocation: list[int] | None = [n1, n2]
        assumptions.append("Group allocation is rounded up while preserving the requested ratio.")
    else:
        nobs = float(
            one_power.solve_power(
                effect_size=effect,
                nobs=None,
                alpha=alpha,
                power=power,
                alternative=alternative,
            )
        )
        recommended_n = max(2, math.ceil(nobs))
        group_allocation = None

    return {
        "test": test,
        "effect_size": effect,
        "alpha": alpha,
        "target_power": power,
        "alternative": alternative,
        "recommended_n": int(recommended_n),
        "group_allocation": group_allocation,
        "assumptions": assumptions,
    }


def power_curve(
    effect_sizes: Sequence[float],
    *,
    n: int,
    test: str,
    alpha: float = 0.05,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Compute achieved power over a sequence of effect sizes."""
    _validate_common_inputs(
        test=test,
        alpha=alpha,
        ratio=ratio,
        alternative=alternative,
    )
    if not effect_sizes:
        raise ValueError("effect_sizes must not be empty.")
    if n <= 1:
        raise ValueError("n must be greater than 1.")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(_POWER_IMPORT_ERROR) from exc

    rows = [
        {
            "effect_size": abs(float(effect_size)),
            "power": _compute_power(
                abs(float(effect_size)),
                n=n,
                test=test,
                alpha=alpha,
                ratio=ratio,
                alternative=alternative,
            ),
        }
        for effect_size in effect_sizes
    ]
    return pd.DataFrame(rows, columns=["effect_size", "power"])


def minimum_detectable_effect(
    n: int,
    *,
    test: str,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Solve for the smallest detectable standardized effect size."""
    _validate_common_inputs(
        test=test,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative,
    )
    if n <= 1:
        raise ValueError("n must be greater than 1.")

    low = 1e-6
    high = 5.0
    if (
        _compute_power(high, n=n, test=test, alpha=alpha, ratio=ratio, alternative=alternative)
        < power
    ):
        raise ValueError("Target power is unreachable for effect sizes up to 5.0.")

    for _ in range(100):
        mid = (low + high) / 2.0
        achieved = _compute_power(
            mid,
            n=n,
            test=test,
            alpha=alpha,
            ratio=ratio,
            alternative=alternative,
        )
        if achieved >= power:
            high = mid
        else:
            low = mid
        if abs(high - low) < 1e-4:
            break

    return {
        "test": test,
        "n": int(n),
        "alpha": alpha,
        "target_power": power,
        "alternative": alternative,
        "minimum_detectable_effect": float(high),
        "assumptions": ["Effect size is interpreted as Cohen's d."],
    }


__all__ = [
    "ConditionComparisonReport",
    "ConditionPairComparison",
    "GroupComparisonResult",
    "MixedEffectsResult",
    "RegressionResult",
    "bootstrap_ci",
    "build_condition_metric_table",
    "compare_condition_pairs",
    "compare_groups",
    "estimate_sample_size",
    "fit_mixed_effects",
    "fit_regression",
    "minimum_detectable_effect",
    "permutation_test",
    "power_curve",
    "rank_tests_one_stop",
]
