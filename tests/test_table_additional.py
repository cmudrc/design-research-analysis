from __future__ import annotations

from datetime import UTC, datetime

import pytest

import design_research_analysis.table as table_module
from design_research_analysis.table import (
    UnifiedTableConfig,
    UnifiedTableValidationReport,
    coerce_unified_table,
    derive_columns,
    group_rows,
    iter_non_blank,
    select_column,
    validate_unified_table,
)


def test_unified_table_config_and_report_serialize() -> None:
    config = UnifiedTableConfig(optional_columns=("meta_json", "notes"))
    report = UnifiedTableValidationReport(
        is_valid=False,
        n_rows=2,
        columns=("timestamp", "notes"),
        missing_required=("record_id",),
        missing_recommended=("text",),
        errors=("bad timestamp",),
        warnings=("missing text",),
    )

    assert config.known_columns() == {
        "timestamp",
        "record_id",
        "text",
        "session_id",
        "actor_id",
        "event_type",
        "meta_json",
        "notes",
    }
    assert report.to_dict()["columns"] == ["timestamp", "notes"]


def test_parse_timestamp_value_variants_and_errors() -> None:
    parsed_epoch = table_module._parse_timestamp_value(0, column="timestamp")
    parsed_naive = table_module._parse_timestamp_value("2026-01-01T10:00:00", column="timestamp")

    assert parsed_epoch.tzinfo == UTC
    assert parsed_naive.tzinfo == UTC

    with pytest.raises(ValueError, match="empty timestamp string"):
        table_module._parse_timestamp_value("   ", column="timestamp")
    with pytest.raises(ValueError, match="invalid ISO timestamp"):
        table_module._parse_timestamp_value("not-a-date", column="timestamp")
    with pytest.raises(ValueError, match="unsupported timestamp type"):
        table_module._parse_timestamp_value(object(), column="timestamp")


def test_rows_from_data_and_coercion_validation_paths() -> None:
    assert table_module._columnar_to_rows({}) == []
    assert table_module._rows_from_data({"timestamp": ["2026-01-01T10:00:00Z"]}) == [
        {"timestamp": "2026-01-01T10:00:00Z"}
    ]

    with pytest.raises(ValueError, match="columnar"):
        table_module._rows_from_data({"timestamp": "2026-01-01T10:00:00Z"})
    with pytest.raises(ValueError, match="mapping-like"):
        table_module._rows_from_data([1, 2, 3])
    with pytest.raises(ValueError, match="Unsupported table input"):
        table_module._rows_from_data("bad-input")

    with pytest.raises(ValueError, match="Failed to parse timestamp"):
        coerce_unified_table([{"timestamp": "not-a-date"}])


def test_coerce_unified_table_sorts_timestamped_rows_and_preserves_trailing_blanks() -> None:
    rows = coerce_unified_table(
        {
            "timestamp": ["2026-01-01T10:00:01Z", "", "2026-01-01T10:00:00Z"],
            "text": ["later", "blank", "earlier"],
        }
    )

    assert rows[0]["text"] == "earlier"
    assert rows[1]["text"] == "later"
    assert rows[2]["text"] == "blank"
    assert isinstance(rows[0]["timestamp"], datetime)


def test_validate_unified_table_reports_empty_extra_blank_and_invalid_rows() -> None:
    empty_report = validate_unified_table([])
    assert empty_report.is_valid is False
    assert "must contain at least one row" in empty_report.errors[0]

    report = validate_unified_table(
        [
            {"timestamp": "", "unexpected": 1},
            {"timestamp": "not-a-date", "record_id": "r2"},
        ],
        config=UnifiedTableConfig(allow_extra_columns=False),
    )

    assert report.is_valid is False
    assert "Unexpected columns: unexpected." in report.errors
    assert "Row 0 has blank timestamp in 'timestamp'." in report.errors
    assert any("Row 1 timestamp is invalid" in error for error in report.errors)
    assert report.missing_recommended
    assert report.warnings


def test_derive_group_select_and_iter_helpers_cover_all_mappers() -> None:
    rows = derive_columns(
        [
            {"timestamp": "2026-01-01T10:00:00Z", "record_id": "", "session_id": None},
            {"timestamp": "2026-01-01T10:00:01Z", "text": "", "event_type": "", "actor_id": ""},
        ],
        record_id_mapper=lambda row: f"rid:{row['timestamp']}",
        session_mapper=lambda row: "session-a",
        actor_mapper=lambda row: "actor-a",
        event_mapper=lambda row: "message",
        text_mapper=lambda row: f"text:{row['timestamp']}",
    )

    assert rows[0]["record_id"] == "rid:2026-01-01T10:00:00Z"
    assert rows[0]["session_id"] == "session-a"
    assert rows[1]["actor_id"] == "actor-a"
    assert rows[1]["event_type"] == "message"
    assert rows[1]["text"] == "text:2026-01-01T10:00:01Z"

    grouped = group_rows(
        [*rows, {"session_id": None, "record_id": "missing"}],
        key_column="session_id",
    )
    assert grouped[0][0]["record_id"] == "missing"
    assert grouped[1][0]["session_id"] == "session-a"

    assert select_column(rows, "record_id") == [
        "rid:2026-01-01T10:00:00Z",
        "rid:2026-01-01T10:00:01Z",
    ]
    assert list(iter_non_blank([None, "", " value ", 0])) == [" value ", 0]
