"""Unified table contracts for design-research analysis workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

Row = dict[str, Any]
MapperFn = Callable[[Mapping[str, Any]], Any]

_RECOMMENDED_COLUMNS = ("record_id", "text", "session_id", "actor_id", "event_type")
_OPTIONAL_COLUMNS = ("meta_json",)


@dataclass(frozen=True, slots=True)
class UnifiedTableConfig:
    """Configuration for coercing and validating a unified table.

    Args:
        required_columns: Columns that must be present.
        recommended_columns: Columns that are strongly encouraged.
        optional_columns: Common optional fields documented by the package.
        timestamp_column: Name of the canonical timestamp column.
        parse_timestamps: Whether to parse timestamp values into ``datetime`` objects.
        sort_by_timestamp: Whether to return rows sorted by timestamp.
        allow_extra_columns: Whether columns outside known sets are allowed.
    """

    required_columns: tuple[str, ...] = ("timestamp",)
    recommended_columns: tuple[str, ...] = _RECOMMENDED_COLUMNS
    optional_columns: tuple[str, ...] = _OPTIONAL_COLUMNS
    timestamp_column: str = "timestamp"
    parse_timestamps: bool = True
    sort_by_timestamp: bool = True
    allow_extra_columns: bool = True

    def known_columns(self) -> set[str]:
        """Return the known column names implied by this configuration."""
        return (
            set(self.required_columns)
            | set(self.recommended_columns)
            | set(self.optional_columns)
        )


@dataclass(frozen=True, slots=True)
class UnifiedTableValidationReport:
    """Validation report for a unified table.

    Args:
        is_valid: Whether validation passed.
        n_rows: Number of rows observed.
        columns: Ordered columns found in the table.
        missing_required: Required columns missing from the table.
        missing_recommended: Recommended columns missing from the table.
        errors: Validation errors.
        warnings: Validation warnings.
    """

    is_valid: bool
    n_rows: int
    columns: tuple[str, ...]
    missing_required: tuple[str, ...]
    missing_recommended: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the report."""
        return {
            "is_valid": self.is_valid,
            "n_rows": int(self.n_rows),
            "columns": list(self.columns),
            "missing_required": list(self.missing_required),
            "missing_recommended": list(self.missing_recommended),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _parse_timestamp_value(value: Any, *, column: str) -> datetime:
    """Parse a timestamp value into a timezone-aware ``datetime``.

    Args:
        value: Raw timestamp value.
        column: Name of the timestamp column for error messages.

    Returns:
        Parsed ``datetime``.

    Raises:
        ValueError: If parsing fails.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=UTC)
    elif isinstance(value, str):
        text = value.strip()
        if text == "":
            raise ValueError(f"{column} contains an empty timestamp string.")
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{column} has an invalid ISO timestamp: {value!r}.") from exc
    else:
        raise ValueError(f"{column} has an unsupported timestamp type: {type(value)!r}.")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _columnar_to_rows(data: Mapping[str, Sequence[Any]]) -> list[Row]:
    lengths = {len(values) for values in data.values()}
    if not lengths:
        return []
    if len(lengths) != 1:
        raise ValueError("Columnar input must provide equally sized value arrays.")
    size = lengths.pop()
    rows: list[Row] = []
    for idx in range(size):
        rows.append({column: values[idx] for column, values in data.items()})
    return rows


def _rows_from_data(data: Any) -> list[Row]:
    if isinstance(data, Mapping):
        if data and all(
            isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            for value in data.values()
        ):
            return _columnar_to_rows(data)
        raise ValueError(
            "Mapping input must be columnar (column name -> sequence of values) "
            "to coerce a unified table."
        )

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        rows: list[Row] = []
        for index, row in enumerate(data):
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"Row input must be mapping-like. Found {type(row)!r} at index {index}."
                )
            rows.append(dict(row))
        return rows

    raise ValueError(
        "Unsupported table input. Provide row-oriented data (sequence of mappings) "
        "or columnar data (mapping of column -> sequence)."
    )


def _stable_columns(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        for column in row:
            if column not in seen:
                seen.add(column)
                ordered.append(column)
    return tuple(ordered)


def coerce_unified_table(
    data: Any,
    *,
    config: UnifiedTableConfig | None = None,
) -> list[Row]:
    """Coerce input data to normalized row-oriented unified table records.

    Args:
        data: Row-oriented sequence of mappings or column-oriented mapping.
        config: Optional table configuration.

    Returns:
        Normalized table rows.
    """
    resolved_config = config or UnifiedTableConfig()
    rows = _rows_from_data(data)

    if not rows:
        return []

    normalized: list[Row] = [dict(row) for row in rows]

    if resolved_config.parse_timestamps:
        ts_column = resolved_config.timestamp_column
        for index, row in enumerate(normalized):
            if ts_column in row and not _is_blank(row[ts_column]):
                try:
                    row[ts_column] = _parse_timestamp_value(row[ts_column], column=ts_column)
                except ValueError as exc:
                    raise ValueError(f"Failed to parse timestamp at row {index}: {exc}") from exc

    if (
        resolved_config.sort_by_timestamp
        and resolved_config.timestamp_column in _stable_columns(normalized)
    ):
        ts_column = resolved_config.timestamp_column
        decorated: list[tuple[int, datetime, Row]] = []
        for index, row in enumerate(normalized):
            raw = row.get(ts_column)
            if _is_blank(raw):
                continue
            if isinstance(raw, datetime):
                dt = raw
            else:
                dt = _parse_timestamp_value(raw, column=ts_column)
                row[ts_column] = dt
            decorated.append((index, dt, row))

        if decorated:
            present_rows = {id(item[2]) for item in decorated}
            sorted_rows = [
                row
                for _, _, row in sorted(decorated, key=lambda item: (item[1], item[0]))
            ]
            trailing = [row for row in normalized if id(row) not in present_rows]
            normalized = sorted_rows + trailing

    return normalized


def validate_unified_table(
    table: Sequence[Mapping[str, Any]],
    *,
    config: UnifiedTableConfig | None = None,
) -> UnifiedTableValidationReport:
    """Validate a unified table against the configured contract.

    Args:
        table: Coerced unified table rows.
        config: Optional table configuration.

    Returns:
        Validation report with errors and warnings.
    """
    resolved_config = config or UnifiedTableConfig()
    rows = [dict(row) for row in table]
    columns = _stable_columns(rows)
    column_set = set(columns)

    errors: list[str] = []
    warnings: list[str] = []

    if not rows:
        errors.append("Unified table must contain at least one row.")

    missing_required = tuple(
        sorted(
            column
            for column in resolved_config.required_columns
            if column not in column_set
        )
    )
    if missing_required:
        errors.append(f"Missing required columns: {', '.join(missing_required)}.")

    missing_recommended = tuple(
        sorted(
            column
            for column in resolved_config.recommended_columns
            if column not in column_set
        )
    )
    if missing_recommended:
        warnings.append(f"Missing recommended columns: {', '.join(missing_recommended)}.")

    if not resolved_config.allow_extra_columns:
        extra = sorted(
            column
            for column in column_set
            if column not in resolved_config.known_columns()
        )
        if extra:
            errors.append(f"Unexpected columns: {', '.join(extra)}.")

    ts_column = resolved_config.timestamp_column
    if ts_column in column_set:
        for index, row in enumerate(rows):
            value = row.get(ts_column)
            if _is_blank(value):
                errors.append(f"Row {index} has blank timestamp in '{ts_column}'.")
                continue
            if resolved_config.parse_timestamps:
                try:
                    _parse_timestamp_value(value, column=ts_column)
                except ValueError as exc:
                    errors.append(f"Row {index} timestamp is invalid: {exc}")

    is_valid = len(errors) == 0
    return UnifiedTableValidationReport(
        is_valid=is_valid,
        n_rows=len(rows),
        columns=columns,
        missing_required=missing_required,
        missing_recommended=missing_recommended,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def derive_columns(
    table: Sequence[Mapping[str, Any]],
    *,
    actor_mapper: MapperFn | None = None,
    event_mapper: MapperFn | None = None,
    session_mapper: MapperFn | None = None,
    text_mapper: MapperFn | None = None,
    record_id_mapper: MapperFn | None = None,
) -> list[Row]:
    """Derive canonical columns from deterministic user-provided mappers.

    Existing non-blank values are preserved. Mappers are only applied to blank or
    missing values. ``record_id`` defaults to the row index if not provided.

    Args:
        table: Unified table rows.
        actor_mapper: Optional mapper for ``actor_id``.
        event_mapper: Optional mapper for ``event_type``.
        session_mapper: Optional mapper for ``session_id``.
        text_mapper: Optional mapper for ``text``.
        record_id_mapper: Optional mapper for ``record_id``.

    Returns:
        New rows with derived columns.
    """
    derived: list[Row] = []
    for index, source in enumerate(table):
        row = dict(source)

        if _is_blank(row.get("record_id")):
            if record_id_mapper is not None:
                row["record_id"] = record_id_mapper(source)
            else:
                row["record_id"] = str(index)

        if _is_blank(row.get("session_id")) and session_mapper is not None:
            row["session_id"] = session_mapper(source)

        if _is_blank(row.get("actor_id")) and actor_mapper is not None:
            row["actor_id"] = actor_mapper(source)

        if _is_blank(row.get("event_type")) and event_mapper is not None:
            row["event_type"] = event_mapper(source)

        if _is_blank(row.get("text")) and text_mapper is not None:
            row["text"] = text_mapper(source)

        derived.append(row)

    return derived


def group_rows(
    table: Sequence[Mapping[str, Any]],
    *,
    key_column: str,
) -> list[list[Row]]:
    """Group rows by ``key_column`` while preserving in-group order.

    Args:
        table: Unified table rows.
        key_column: Column used for grouping.

    Returns:
        List of grouped row lists sorted by key string representation.
    """
    grouped: dict[str, list[Row]] = {}
    for row in table:
        key_raw = row.get(key_column)
        if _is_blank(key_raw):
            key = "__missing__"
        else:
            key = str(key_raw)
        grouped.setdefault(key, []).append(dict(row))
    return [grouped[key] for key in sorted(grouped)]


def select_column(table: Sequence[Mapping[str, Any]], column: str) -> list[Any]:
    """Collect a single column from unified table rows."""
    return [row.get(column) for row in table]


def iter_non_blank(values: Iterable[Any]) -> Iterable[Any]:
    """Yield non-blank values from an iterable."""
    for value in values:
        if not _is_blank(value):
            yield value


__all__ = [
    "MapperFn",
    "UnifiedTableConfig",
    "UnifiedTableValidationReport",
    "coerce_unified_table",
    "derive_columns",
    "group_rows",
    "iter_non_blank",
    "select_column",
    "validate_unified_table",
]
