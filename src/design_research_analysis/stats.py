"""Statistical analysis utilities for unified design-research workflows."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ._comparison import ComparableResultMixin
from .table import coerce_unified_table

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
    if alternative not in {"two-sided", "greater", "less"}:
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
    "GroupComparisonResult",
    "MixedEffectsResult",
    "RegressionResult",
    "bootstrap_ci",
    "compare_groups",
    "estimate_sample_size",
    "fit_mixed_effects",
    "fit_regression",
    "minimum_detectable_effect",
    "permutation_test",
    "power_curve",
    "rank_tests_one_stop",
]
