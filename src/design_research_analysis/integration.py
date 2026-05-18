"""Helpers for consuming canonical experiment exports."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ._comparison import ComparisonResult
from .sequence import MarkovChainResult, fit_markov_chain_from_table
from .stats import (
    ConditionComparisonReport,
    RegressionResult,
    build_condition_metric_table,
    compare_condition_pairs,
    fit_regression,
)
from .table import UnifiedTableValidationReport, coerce_unified_table, validate_unified_table

_ANALYSIS_ARTIFACT_FILES = (
    "manifest.json",
    "conditions.csv",
    "runs.csv",
    "events.csv",
    "evaluations.csv",
)
_CONDITION_CONTEXT_EXCLUDE_COLUMNS = frozenset(
    {
        "study_id",
        "condition_id",
        "admissible",
        "constraint_messages",
        "assignment_meta_json",
    }
)
_RUN_CONTEXT_EXCLUDE_COLUMNS = frozenset({"trace_path", "manifest_path"})


def load_experiment_artifacts(path: str | Path) -> dict[str, Any]:
    """Load the canonical analysis-facing experiment artifacts.

    Args:
        path: Study output directory or the canonical ``events.csv`` path inside it.

    Returns:
        Mapping keyed by canonical artifact filename.

    Raises:
        ValueError: If ``path`` does not resolve to a canonical artifact directory.
    """
    return _load_artifact_rows(path, artifact_names=_ANALYSIS_ARTIFACT_FILES)


def validate_experiment_events(path: str | Path) -> UnifiedTableValidationReport:
    """Validate canonical ``events.csv`` output from design-research-experiments.

    Args:
        path: Study output directory or the canonical ``events.csv`` path inside it.

    Returns:
        Unified-table validation report for the exported event rows.

    Raises:
        ValueError: If ``path`` does not resolve to a canonical ``events.csv`` artifact.
    """
    events_path = _resolve_events_path(path)
    rows = _read_csv(events_path)
    return validate_unified_table(coerce_unified_table(rows))


def build_condition_metric_table_from_artifacts(
    path: str | Path,
    *,
    metric: str,
    condition_column: str = "condition",
    run_id_column: str = "run_id",
    condition_id_column: str = "condition_id",
    evaluation_metric_column: str = "metric_name",
    evaluation_value_column: str = "metric_value",
) -> list[dict[str, Any]]:
    """Build a condition metric table directly from canonical experiment artifacts."""
    artifacts = _load_artifact_rows(
        path,
        artifact_names=("conditions.csv", "runs.csv", "evaluations.csv"),
    )
    return build_condition_metric_table(
        artifacts["runs.csv"],
        metric=metric,
        condition_column=condition_column,
        evaluations=artifacts["evaluations.csv"],
        conditions=artifacts["conditions.csv"],
        run_id_column=run_id_column,
        condition_id_column=condition_id_column,
        evaluation_metric_column=evaluation_metric_column,
        evaluation_value_column=evaluation_value_column,
    )


def compare_condition_pairs_from_artifacts(
    path: str | Path,
    *,
    metric: str,
    condition_column: str = "condition",
    condition_pairs: Sequence[tuple[str, str]] | None = None,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    exact_threshold: int = 250_000,
    n_permutations: int = 20_000,
    seed: int = 0,
) -> ConditionComparisonReport:
    """Compare condition pairs directly from canonical experiment artifacts."""
    rows = build_condition_metric_table_from_artifacts(
        path,
        metric=metric,
        condition_column=condition_column,
    )
    return compare_condition_pairs(
        rows,
        condition_pairs=condition_pairs,
        alternative=alternative,
        alpha=alpha,
        exact_threshold=exact_threshold,
        n_permutations=n_permutations,
        seed=seed,
    )


def build_event_table_from_artifacts(
    path: str | Path,
    *,
    condition_columns: Sequence[str] | None = None,
    run_columns: Sequence[str] | None = None,
    run_id_column: str = "run_id",
    session_column: str = "session_id",
    condition_id_column: str = "condition_id",
) -> list[dict[str, Any]]:
    """Return event rows enriched with run and condition context from artifacts."""
    artifacts = _load_artifact_rows(
        path,
        artifact_names=("conditions.csv", "runs.csv", "events.csv"),
    )
    conditions = _rows(artifacts["conditions.csv"], table_name="conditions.csv")
    runs = _rows(artifacts["runs.csv"], table_name="runs.csv")
    events = _rows(artifacts["events.csv"], table_name="events.csv")

    condition_lookup = _unique_row_map(
        conditions,
        key_column=condition_id_column,
        table_name="conditions.csv",
    )
    run_lookup = _unique_row_map(runs, key_column=run_id_column, table_name="runs.csv")
    resolved_condition_columns = _resolve_context_columns(
        conditions,
        requested_columns=condition_columns,
        exclude_columns=_CONDITION_CONTEXT_EXCLUDE_COLUMNS,
        table_name="conditions.csv",
    )
    resolved_run_columns = _resolve_context_columns(
        runs,
        requested_columns=run_columns,
        exclude_columns=_RUN_CONTEXT_EXCLUDE_COLUMNS,
        table_name="runs.csv",
    )

    enriched: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        row = dict(event)
        run_row = _resolve_event_run(
            row,
            run_lookup,
            row_index=index,
            run_id_column=run_id_column,
            session_column=session_column,
        )
        condition_row = _resolve_condition(
            row,
            run_row,
            condition_lookup,
            row_index=index,
            condition_id_column=condition_id_column,
        )

        _merge_context_columns(row, run_row, resolved_run_columns, source="run")
        _merge_context_columns(
            row,
            condition_row,
            resolved_condition_columns,
            source="condition",
        )
        enriched.append(row)

    return enriched


def fit_markov_chains_from_artifacts(
    path: str | Path,
    *,
    condition_column: str,
    order: int = 1,
    smoothing: float = 1.0,
    event_column: str = "event_type",
    session_column: str = "run_id",
    actor_column: str = "actor_id",
    include_actor_in_token: bool = False,
) -> dict[str, MarkovChainResult]:
    """Fit one Markov chain per condition directly from experiment artifacts."""
    events = build_event_table_from_artifacts(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(events):
        condition = row.get(condition_column)
        if _is_blank(condition):
            raise ValueError(
                f"events row {index} is missing condition column {condition_column!r}."
            )
        grouped[str(condition)].append(row)

    if len(grouped) < 2:
        raise ValueError("At least two condition groups are required.")

    results: dict[str, MarkovChainResult] = {}
    for condition, rows in sorted(grouped.items()):
        result = fit_markov_chain_from_table(
            rows,
            order=order,
            smoothing=smoothing,
            event_column=event_column,
            session_column=session_column,
            actor_column=actor_column,
            include_actor_in_token=include_actor_in_token,
        )
        result.config.update(
            {
                "source": "experiment_artifacts",
                "condition_column": condition_column,
                "condition": condition,
            }
        )
        results[condition] = result
    return results


def compare_markov_chains_from_artifacts(
    path: str | Path,
    *,
    condition_column: str,
    left_condition: str,
    right_condition: str,
    order: int = 1,
    smoothing: float = 1.0,
    event_column: str = "event_type",
    session_column: str = "run_id",
    actor_column: str = "actor_id",
    include_actor_in_token: bool = False,
) -> ComparisonResult:
    """Compare two condition-specific Markov chains from experiment artifacts."""
    chains = fit_markov_chains_from_artifacts(
        path,
        condition_column=condition_column,
        order=order,
        smoothing=smoothing,
        event_column=event_column,
        session_column=session_column,
        actor_column=actor_column,
        include_actor_in_token=include_actor_in_token,
    )
    try:
        return chains[str(left_condition)] - chains[str(right_condition)]
    except KeyError as exc:
        available = ", ".join(sorted(chains))
        raise ValueError(
            f"Unknown condition {exc.args[0]!r}. Available conditions: {available}."
        ) from exc


def build_run_metric_table_from_artifacts(
    path: str | Path,
    *,
    metrics: str | Sequence[str],
    condition_columns: Sequence[str] | None = None,
    run_columns: Sequence[str] | None = None,
    run_id_column: str = "run_id",
    condition_id_column: str = "condition_id",
    evaluation_metric_column: str = "metric_name",
    evaluation_value_column: str = "metric_value",
) -> list[dict[str, Any]]:
    """Return one run-level row with requested metrics and experiment context."""
    artifacts = _load_artifact_rows(
        path,
        artifact_names=("conditions.csv", "runs.csv", "evaluations.csv"),
    )
    return _build_run_metric_table(
        artifacts,
        metrics=_as_name_list(metrics, name="metrics"),
        condition_columns=condition_columns,
        run_columns=run_columns,
        run_id_column=run_id_column,
        condition_id_column=condition_id_column,
        evaluation_metric_column=evaluation_metric_column,
        evaluation_value_column=evaluation_value_column,
    )


def fit_regression_from_artifacts(
    path: str | Path,
    *,
    outcome: str,
    predictors: Sequence[str],
    categorical_predictors: Sequence[str] = (),
    condition_columns: Sequence[str] | None = None,
    run_columns: Sequence[str] | None = None,
    add_intercept: bool = True,
    drop_first: bool = True,
) -> RegressionResult:
    """Fit ordinary least squares regression from canonical experiment artifacts."""
    if not predictors:
        raise ValueError("predictors must contain at least one column.")

    artifacts = _load_artifact_rows(
        path,
        artifact_names=("conditions.csv", "runs.csv", "evaluations.csv"),
    )
    context_columns = _artifact_context_columns(
        artifacts,
        condition_columns=condition_columns,
        run_columns=run_columns,
    )
    metric_names = _artifact_metric_names(artifacts)
    metric_columns = [outcome]
    for predictor in predictors:
        if predictor not in context_columns and predictor in metric_names:
            metric_columns.append(predictor)

    rows = _build_run_metric_table(
        artifacts,
        metrics=_dedupe(metric_columns),
        condition_columns=condition_columns,
        run_columns=run_columns,
    )
    matrix, response, feature_names = _encode_regression_rows(
        rows,
        outcome=outcome,
        predictors=predictors,
        categorical_predictors=categorical_predictors,
        drop_first=drop_first,
    )
    result = fit_regression(
        matrix,
        response,
        feature_names=feature_names,
        add_intercept=add_intercept,
    )
    result.config.update(
        {
            "source": "experiment_artifacts",
            "outcome": outcome,
            "predictors": list(predictors),
            "categorical_predictors": list(categorical_predictors),
            "drop_first": bool(drop_first),
        }
    )
    return result


def _resolve_output_dir(path: str | Path) -> Path:
    """Resolve one study output directory from a directory or events path."""
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        return candidate
    if candidate.is_file() and candidate.name == "events.csv":
        return candidate.parent
    raise ValueError(
        "Expected a study output directory or the canonical 'events.csv' artifact path."
    )


def _resolve_events_path(path: str | Path) -> Path:
    """Resolve the canonical ``events.csv`` path from a directory or file input."""
    candidate = Path(path).expanduser()
    events_path = candidate / "events.csv" if candidate.is_dir() else candidate

    if not events_path.is_file() or events_path.name != "events.csv":
        raise ValueError(
            "Expected a study output directory or the canonical 'events.csv' artifact path."
        )
    return events_path


def _load_artifact_rows(
    path: str | Path,
    *,
    artifact_names: Sequence[str],
) -> dict[str, Any]:
    """Load the requested canonical artifacts from one export directory."""
    output_dir = _resolve_output_dir(path)
    _require_artifacts(output_dir, artifact_names)
    rows: dict[str, Any] = {}
    for artifact_name in artifact_names:
        artifact_path = output_dir / artifact_name
        rows[artifact_name] = (
            _read_json(artifact_path)
            if artifact_path.suffix.lower() == ".json"
            else _read_csv(artifact_path)
        )
    return rows


def _require_artifacts(output_dir: Path, artifact_names: Sequence[str]) -> None:
    """Raise a clear error when a canonical export file is missing."""
    missing = [name for name in artifact_names if not (output_dir / name).exists()]
    if missing:
        raise ValueError("Missing canonical experiment artifacts: " + ", ".join(missing) + ".")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read one CSV artifact into row dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON artifact into a dictionary."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object payload in '{path.name}'.")
    return payload


def _rows(data: Any, *, table_name: str) -> list[dict[str, Any]]:
    try:
        rows = coerce_unified_table(data, config=None)
    except ValueError as exc:
        raise ValueError(f"Failed to coerce {table_name}: {exc}") from exc
    return rows


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _as_name_list(names: str | Sequence[str], *, name: str) -> list[str]:
    if isinstance(names, str):
        if not names.strip():
            raise ValueError(f"{name} must not contain blank column names.")
        return [names]
    resolved = [str(item) for item in names]
    if not resolved or any(not item.strip() for item in resolved):
        raise ValueError(f"{name} must not be empty or contain blank column names.")
    return resolved


def _dedupe(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def _stable_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(str(column))
                seen.add(str(column))
    return columns


def _resolve_context_columns(
    rows: Sequence[Mapping[str, Any]],
    *,
    requested_columns: Sequence[str] | None,
    exclude_columns: frozenset[str],
    table_name: str,
) -> list[str]:
    available = _stable_columns(rows)
    if requested_columns is None:
        return [column for column in available if column not in exclude_columns]

    resolved = _as_name_list(tuple(requested_columns), name="requested_columns")
    missing = [column for column in resolved if column not in available]
    if missing:
        raise ValueError(f"{table_name} is missing requested columns: {', '.join(missing)}.")
    return resolved


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
            raise ValueError(f"{table_name} row {index} is missing {key_column!r}.")
        key = str(raw_key)
        if key in resolved:
            raise ValueError(f"{table_name} contains duplicate {key_column!r} value {key!r}.")
        resolved[key] = dict(row)
    return resolved


def _resolve_event_run(
    event: Mapping[str, Any],
    run_lookup: Mapping[str, dict[str, Any]],
    *,
    row_index: int,
    run_id_column: str,
    session_column: str,
) -> dict[str, Any]:
    raw_run_id = event.get(run_id_column)
    if _is_blank(raw_run_id):
        raw_run_id = event.get(session_column)
    if _is_blank(raw_run_id):
        raise ValueError(
            f"events.csv row {row_index} is missing {run_id_column!r} and {session_column!r}."
        )
    run_id = str(raw_run_id)
    try:
        return run_lookup[run_id]
    except KeyError as exc:
        raise ValueError(
            f"events.csv row {row_index} references unknown run_id {run_id!r}."
        ) from exc


def _resolve_condition(
    event: Mapping[str, Any],
    run_row: Mapping[str, Any],
    condition_lookup: Mapping[str, dict[str, Any]],
    *,
    row_index: int,
    condition_id_column: str,
) -> dict[str, Any]:
    raw_condition_id = event.get(condition_id_column)
    if _is_blank(raw_condition_id):
        raw_condition_id = run_row.get(condition_id_column)
    if _is_blank(raw_condition_id):
        raise ValueError(f"events.csv row {row_index} has no {condition_id_column!r} context.")
    condition_id = str(raw_condition_id)
    try:
        return condition_lookup[condition_id]
    except KeyError as exc:
        raise ValueError(
            f"events.csv row {row_index} references unknown condition_id {condition_id!r}."
        ) from exc


def _merge_context_columns(
    target: dict[str, Any],
    source_row: Mapping[str, Any],
    columns: Sequence[str],
    *,
    source: str,
) -> None:
    for column in columns:
        if column not in source_row:
            continue
        value = source_row[column]
        if column not in target or _is_blank(target[column]):
            target[column] = value
        elif not _is_blank(value) and str(target[column]) != str(value):
            target[f"{source}_{column}"] = value


def _artifact_context_columns(
    artifacts: Mapping[str, Any],
    *,
    condition_columns: Sequence[str] | None,
    run_columns: Sequence[str] | None,
) -> set[str]:
    conditions = _rows(artifacts["conditions.csv"], table_name="conditions.csv")
    runs = _rows(artifacts["runs.csv"], table_name="runs.csv")
    return set(
        _resolve_context_columns(
            conditions,
            requested_columns=condition_columns,
            exclude_columns=_CONDITION_CONTEXT_EXCLUDE_COLUMNS,
            table_name="conditions.csv",
        )
    ) | set(
        _resolve_context_columns(
            runs,
            requested_columns=run_columns,
            exclude_columns=_RUN_CONTEXT_EXCLUDE_COLUMNS,
            table_name="runs.csv",
        )
    )


def _artifact_metric_names(artifacts: Mapping[str, Any]) -> set[str]:
    runs = _rows(artifacts["runs.csv"], table_name="runs.csv")
    evaluations = _rows(artifacts["evaluations.csv"], table_name="evaluations.csv")
    names = set(_stable_columns(runs))
    for row in evaluations:
        metric_name = row.get("metric_name")
        if not _is_blank(metric_name):
            names.add(str(metric_name))
    return names


def _build_run_metric_table(
    artifacts: Mapping[str, Any],
    *,
    metrics: Sequence[str],
    condition_columns: Sequence[str] | None,
    run_columns: Sequence[str] | None,
    run_id_column: str = "run_id",
    condition_id_column: str = "condition_id",
    evaluation_metric_column: str = "metric_name",
    evaluation_value_column: str = "metric_value",
) -> list[dict[str, Any]]:
    conditions = _rows(artifacts["conditions.csv"], table_name="conditions.csv")
    runs = _rows(artifacts["runs.csv"], table_name="runs.csv")
    evaluations = _rows(artifacts["evaluations.csv"], table_name="evaluations.csv")
    condition_lookup = _unique_row_map(
        conditions,
        key_column=condition_id_column,
        table_name="conditions.csv",
    )
    evaluation_lookup = _collect_rows_by_run_id(
        evaluations,
        run_id_column=run_id_column,
        table_name="evaluations.csv",
    )
    resolved_condition_columns = _resolve_context_columns(
        conditions,
        requested_columns=condition_columns,
        exclude_columns=_CONDITION_CONTEXT_EXCLUDE_COLUMNS,
        table_name="conditions.csv",
    )
    resolved_run_columns = _resolve_context_columns(
        runs,
        requested_columns=run_columns,
        exclude_columns=_RUN_CONTEXT_EXCLUDE_COLUMNS,
        table_name="runs.csv",
    )

    rows: list[dict[str, Any]] = []
    for index, run_row in enumerate(runs):
        raw_run_id = run_row.get(run_id_column)
        if _is_blank(raw_run_id):
            raise ValueError(f"runs.csv row {index} is missing {run_id_column!r}.")
        run_id = str(raw_run_id)
        raw_condition_id = run_row.get(condition_id_column)
        if _is_blank(raw_condition_id):
            raise ValueError(f"runs.csv row {index} is missing {condition_id_column!r}.")
        condition_id = str(raw_condition_id)
        condition_row = condition_lookup.get(condition_id)
        if condition_row is None:
            raise ValueError(
                f"runs.csv row {index} references unknown condition_id {condition_id!r}."
            )

        row: dict[str, Any] = {}
        _merge_context_columns(row, run_row, resolved_run_columns, source="run")
        _merge_context_columns(row, condition_row, resolved_condition_columns, source="condition")
        row.setdefault(run_id_column, run_id)
        row.setdefault(condition_id_column, condition_id)
        for metric in metrics:
            row[metric] = _run_metric_value(
                run_row,
                evaluation_lookup.get(run_id, []),
                metric=metric,
                run_id=run_id,
                evaluation_metric_column=evaluation_metric_column,
                evaluation_value_column=evaluation_value_column,
            )
        rows.append(row)
    return rows


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
            raise ValueError(f"{table_name} row {index} is missing {run_id_column!r}.")
        grouped[str(raw_run_id)].append(dict(row))
    return dict(grouped)


def _run_metric_value(
    run_row: Mapping[str, Any],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    run_id: str,
    evaluation_metric_column: str,
    evaluation_value_column: str,
) -> float:
    if metric in run_row and not _is_blank(run_row[metric]):
        return float(run_row[metric])

    matching = [
        row
        for row in evaluation_rows
        if str(row.get(evaluation_metric_column, "")) == metric
        and (_is_blank(row.get("aggregation_level")) or str(row.get("aggregation_level")) == "run")
    ]
    if not matching:
        raise ValueError(f"No metric {metric!r} was found for run_id {run_id!r}.")
    if len(matching) > 1:
        raise ValueError(f"Multiple metric rows matched {metric!r} for run_id {run_id!r}.")
    value = matching[0].get(evaluation_value_column)
    if _is_blank(value):
        raise ValueError(f"Metric {metric!r} for run_id {run_id!r} is missing a value.")
    return float(cast(float | int | str, value))


def _encode_regression_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome: str,
    predictors: Sequence[str],
    categorical_predictors: Sequence[str],
    drop_first: bool,
) -> tuple[list[list[float]], list[float], list[str]]:
    categorical = set(categorical_predictors)
    unknown_categorical = sorted(categorical.difference(predictors))
    if unknown_categorical:
        raise ValueError(
            "categorical_predictors must be a subset of predictors. Unknown: "
            + ", ".join(unknown_categorical)
        )

    levels_by_predictor: dict[str, list[str]] = {}
    for predictor in categorical:
        levels = sorted(
            {
                str(row[predictor])
                for index, row in enumerate(rows)
                if _require_value(row, predictor, row_index=index) is not None
            }
        )
        if not levels:
            raise ValueError(f"Categorical predictor {predictor!r} has no observed levels.")
        levels_by_predictor[predictor] = levels[1:] if drop_first else levels

    feature_names: list[str] = []
    for predictor in predictors:
        if predictor in categorical:
            feature_names.extend(
                f"{predictor}[{level}]" for level in levels_by_predictor[predictor]
            )
        else:
            feature_names.append(predictor)

    matrix: list[list[float]] = []
    response: list[float] = []
    for row_index, row in enumerate(rows):
        response.append(float(_require_value(row, outcome, row_index=row_index)))
        features: list[float] = []
        for predictor in predictors:
            value = _require_value(row, predictor, row_index=row_index)
            if predictor in categorical:
                encoded_levels = levels_by_predictor[predictor]
                features.extend(1.0 if str(value) == level else 0.0 for level in encoded_levels)
            else:
                try:
                    features.append(float(value))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Predictor {predictor!r} must be numeric or listed in "
                        "categorical_predictors."
                    ) from exc
        matrix.append(features)

    return matrix, response, feature_names


def _require_value(row: Mapping[str, Any], column: str, *, row_index: int) -> Any:
    if column not in row or _is_blank(row[column]):
        raise ValueError(f"analysis row {row_index} is missing {column!r}.")
    return row[column]


__all__ = [
    "build_condition_metric_table_from_artifacts",
    "build_event_table_from_artifacts",
    "build_run_metric_table_from_artifacts",
    "compare_condition_pairs_from_artifacts",
    "compare_markov_chains_from_artifacts",
    "fit_markov_chains_from_artifacts",
    "fit_regression_from_artifacts",
    "load_experiment_artifacts",
    "validate_experiment_events",
]
