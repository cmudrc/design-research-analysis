"""Dataset profiling, validation, and codebook helpers."""

from __future__ import annotations

import json
import re
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    import pandas as pd

_DATA_IMPORT_ERROR = (
    "Dataset profiling requires optional dependencies. "
    "Install with `pip install design-research-analysis[data]`."
)
_DATASET_INPUT_ERROR = (
    "Unsupported dataset input format. Use a pandas DataFrame or a .csv, .tsv, or .json path."
)

_ALLOWED_SCHEMA_KEYS = {"dtype", "required", "nullable", "unique", "min", "max", "allowed", "regex"}


class ColumnSchema(TypedDict, total=False):
    """Schema constraints for a single column."""

    dtype: NotRequired[Literal["numeric", "integer", "string", "category", "boolean", "datetime"]]
    required: NotRequired[bool]
    nullable: NotRequired[bool]
    unique: NotRequired[bool]
    min: NotRequired[Any]
    max: NotRequired[Any]
    allowed: NotRequired[Any]
    regex: NotRequired[str]


DataSchema = dict[str, ColumnSchema]


def _load_pandas() -> tuple[Any, Any]:
    try:
        import pandas as pd
        from pandas.api import types as pd_types
    except ImportError as exc:
        raise ImportError(_DATA_IMPORT_ERROR) from exc
    return pd, pd_types


def _coerce_dataframe_input(data: Any, *, pd_module: Any) -> Any:
    if isinstance(data, pd_module.DataFrame):
        return data

    if isinstance(data, (str, PathLike)):
        input_path = Path(data)
        suffix = input_path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            delimiter = "," if suffix == ".csv" else "\t"
            return pd_module.read_csv(input_path, sep=delimiter)

        if suffix == ".json":
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                if payload and not all(isinstance(item, dict) for item in payload):
                    raise ValueError("JSON row input must be a list of objects.")
                return pd_module.DataFrame(payload)
            if isinstance(payload, dict):
                return pd_module.DataFrame(payload)
            raise ValueError("JSON input must be a list of objects or a columnar object.")

        raise ValueError(_DATASET_INPUT_ERROR)

    raise TypeError("Dataset input must be a pandas DataFrame or a .csv, .tsv, or .json path.")


def _infer_dtype(series: Any, *, pd_module: Any, pd_types: Any) -> str:
    if pd_types.is_bool_dtype(series):
        return "boolean"
    if pd_types.is_datetime64_any_dtype(series):
        return "datetime"
    if isinstance(series.dtype, pd_module.CategoricalDtype):
        return "category"
    if pd_types.is_integer_dtype(series):
        return "integer"
    if pd_types.is_numeric_dtype(series):
        return "numeric"
    if pd_types.is_string_dtype(series) or pd_types.is_object_dtype(series):
        return "string"
    return "string"


def _is_profile_numeric(series: Any, *, pd_module: Any, pd_types: Any) -> bool:
    inferred = _infer_dtype(series, pd_module=pd_module, pd_types=pd_types)
    return inferred in {"numeric", "integer"}


def _sorted_stringified_values(series: Any, *, limit: int = 3) -> list[str]:
    unique_values = {str(value) for value in series.dropna().unique().tolist()}
    return sorted(unique_values)[:limit]


def _numeric_summary(series: Any, *, pd_module: Any) -> dict[str, Any]:
    numeric = pd_module.to_numeric(series.dropna(), errors="coerce").dropna()
    outlier_count = 0
    if len(numeric) >= 8:
        q1 = float(numeric.quantile(0.25))
        q3 = float(numeric.quantile(0.75))
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((numeric < lower) | (numeric > upper)).sum())
    return {
        "mean": float(numeric.mean()) if not numeric.empty else None,
        "std": float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
        "min": float(numeric.min()) if not numeric.empty else None,
        "max": float(numeric.max()) if not numeric.empty else None,
        "median": float(numeric.median()) if not numeric.empty else None,
        "outlier_count_iqr": outlier_count,
    }


def profile_dataframe(
    df: pd.DataFrame | str | PathLike[str],
    *,
    max_categorical_levels: int = 20,
) -> dict[str, Any]:
    """Profile a DataFrame or supported dataset file without mutating inputs."""
    pd_module, pd_types = _load_pandas()
    frame = _coerce_dataframe_input(df, pd_module=pd_module)
    warnings: list[str] = []
    column_summaries: dict[str, dict[str, Any]] = {}

    for column_name in frame.columns:
        series = frame[column_name]
        inferred_dtype = _infer_dtype(series, pd_module=pd_module, pd_types=pd_types)
        nonnull = series.dropna()
        n_unique = int(nonnull.nunique(dropna=True))
        summary: dict[str, Any] = {
            "inferred_dtype": inferred_dtype,
            "nonnull_count": int(nonnull.shape[0]),
            "missing_count": int(series.isna().sum()),
            "missing_fraction": float(series.isna().mean()) if len(series) else 0.0,
            "n_unique": n_unique,
        }
        if _is_profile_numeric(series, pd_module=pd_module, pd_types=pd_types):
            summary.update(_numeric_summary(series, pd_module=pd_module))
        else:
            summary["sample_values"] = _sorted_stringified_values(series)
            if n_unique > max_categorical_levels:
                warnings.append(
                    f"Column '{column_name}' has {n_unique} unique values, exceeding the "
                    f"recommended categorical limit of {max_categorical_levels}."
                )
        column_summaries[column_name] = summary

    return {
        "n_rows": len(frame),
        "n_columns": len(frame.columns),
        "columns": column_summaries,
        "warnings": warnings,
    }


def _matches_declared_dtype(series: Any, *, dtype_name: str, pd_module: Any, pd_types: Any) -> bool:
    if dtype_name == "numeric":
        return bool(pd_types.is_numeric_dtype(series) and not pd_types.is_bool_dtype(series))
    if dtype_name == "integer":
        return bool(pd_types.is_integer_dtype(series))
    if dtype_name == "string":
        return bool(pd_types.is_string_dtype(series) or pd_types.is_object_dtype(series))
    if dtype_name == "category":
        return bool(isinstance(series.dtype, pd_module.CategoricalDtype))
    if dtype_name == "boolean":
        return bool(pd_types.is_bool_dtype(series))
    if dtype_name == "datetime":
        return bool(pd_types.is_datetime64_any_dtype(series))
    return False


def validate_dataframe(
    df: pd.DataFrame | str | PathLike[str],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Validate a DataFrame or supported dataset file against a declarative schema."""
    pd_module, pd_types = _load_pandas()
    frame = _coerce_dataframe_input(df, pd_module=pd_module)
    errors: list[str] = []
    warnings: list[str] = []
    checked_columns: list[str] = []
    missing_columns: list[str] = []

    for column_name, raw_rules in schema.items():
        if not isinstance(raw_rules, dict):
            errors.append(f"Schema for column '{column_name}' must be a mapping.")
            continue

        invalid_keys = sorted(set(raw_rules) - _ALLOWED_SCHEMA_KEYS)
        for key in invalid_keys:
            errors.append(f"Column '{column_name}' uses unsupported schema key '{key}'.")

        required = bool(raw_rules.get("required", True))
        if column_name not in frame.columns:
            if required:
                errors.append(f"Required column '{column_name}' is missing.")
                missing_columns.append(column_name)
            continue

        checked_columns.append(column_name)
        series = frame[column_name]
        nonnull = series.dropna()

        declared_dtype = raw_rules.get("dtype")
        if declared_dtype is not None:
            valid_dtypes = {"numeric", "integer", "string", "category", "boolean", "datetime"}
            if declared_dtype not in valid_dtypes:
                errors.append(
                    f"Column '{column_name}' declares unsupported dtype '{declared_dtype}'."
                )
            elif not _matches_declared_dtype(
                series,
                dtype_name=declared_dtype,
                pd_module=pd_module,
                pd_types=pd_types,
            ):
                errors.append(
                    f"Column '{column_name}' does not match declared dtype '{declared_dtype}'."
                )

        if raw_rules.get("nullable") is False and series.isna().any():
            errors.append(f"Column '{column_name}' contains null values but is not nullable.")

        if raw_rules.get("unique") is True and nonnull.duplicated().any():
            errors.append(f"Column '{column_name}' contains duplicate non-null values.")

        if "min" in raw_rules:
            if not (pd_types.is_numeric_dtype(series) or pd_types.is_datetime64_any_dtype(series)):
                errors.append(f"Column '{column_name}' cannot use 'min' for non-numeric data.")
            elif not nonnull.empty and bool((nonnull < raw_rules["min"]).any()):
                errors.append(f"Column '{column_name}' has values below the allowed minimum.")

        if "max" in raw_rules:
            if not (pd_types.is_numeric_dtype(series) or pd_types.is_datetime64_any_dtype(series)):
                errors.append(f"Column '{column_name}' cannot use 'max' for non-numeric data.")
            elif not nonnull.empty and bool((nonnull > raw_rules["max"]).any()):
                errors.append(f"Column '{column_name}' has values above the allowed maximum.")

        if "allowed" in raw_rules:
            allowed_values = raw_rules["allowed"]
            if isinstance(allowed_values, (str, bytes)):
                allowed_lookup = {allowed_values}
            else:
                try:
                    allowed_lookup = set(allowed_values)
                except TypeError:
                    allowed_lookup = {allowed_values}
            if not nonnull.empty and not bool(nonnull.isin(allowed_lookup).all()):
                errors.append(f"Column '{column_name}' contains values outside the allowed set.")

        if "regex" in raw_rules:
            if not (
                pd_types.is_string_dtype(series)
                or pd_types.is_object_dtype(series)
                or isinstance(series.dtype, pd_module.CategoricalDtype)
            ):
                errors.append(f"Column '{column_name}' cannot use 'regex' for non-string data.")
            else:
                pattern = re.compile(str(raw_rules["regex"]))
                matches = nonnull.astype(str).map(
                    lambda value, compiled=pattern: bool(compiled.fullmatch(value))
                )
                if not matches.all():
                    errors.append(f"Column '{column_name}' has values that fail regex validation.")

    unexpected_columns = [column for column in frame.columns if column not in schema]
    for column_name in unexpected_columns:
        warnings.append(f"Column '{column_name}' is not described by the schema.")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "checked_columns": checked_columns,
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
        },
    }


def generate_codebook(
    df: pd.DataFrame | str | PathLike[str],
    *,
    descriptions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Generate a compact codebook from a DataFrame or supported dataset file."""
    pd_module, pd_types = _load_pandas()
    frame = _coerce_dataframe_input(df, pd_module=pd_module)
    rows: list[dict[str, Any]] = []
    description_lookup = descriptions or {}

    for column_name in frame.columns:
        series = frame[column_name]
        nonnull = series.dropna()
        rows.append(
            {
                "column": column_name,
                "inferred_dtype": _infer_dtype(series, pd_module=pd_module, pd_types=pd_types),
                "nonnull_count": int(nonnull.shape[0]),
                "missing_count": int(series.isna().sum()),
                "missing_fraction": float(series.isna().mean()) if len(series) else 0.0,
                "n_unique": int(nonnull.nunique(dropna=True)),
                "example_values": "; ".join(_sorted_stringified_values(series)),
                "description": description_lookup.get(column_name, ""),
            }
        )

    columns = [
        "column",
        "inferred_dtype",
        "nonnull_count",
        "missing_count",
        "missing_fraction",
        "n_unique",
        "example_values",
        "description",
    ]
    return pd_module.DataFrame(rows, columns=columns)


__all__ = [
    "ColumnSchema",
    "DataSchema",
    "generate_codebook",
    "profile_dataframe",
    "validate_dataframe",
]
