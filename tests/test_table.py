from __future__ import annotations

from datetime import UTC, datetime

import pytest

from design_research_analysis.table import (
    UnifiedTableConfig,
    coerce_unified_table,
    derive_columns,
    validate_unified_table,
)


def test_validate_unified_table_requires_timestamp() -> None:
    rows = [{"record_id": "1", "text": "hello"}]
    report = validate_unified_table(rows)
    assert report.is_valid is False
    assert "timestamp" in report.missing_required


def test_validate_unified_table_accepts_loose_minimal_rows() -> None:
    rows = coerce_unified_table([{"timestamp": "2026-01-01T10:00:00Z"}])
    report = validate_unified_table(rows, config=UnifiedTableConfig())

    assert report.is_valid is True
    assert report.n_rows == 1
    assert isinstance(rows[0]["timestamp"], datetime)
    assert rows[0]["timestamp"].tzinfo == UTC


def test_validate_unified_table_accepts_csv_path(tmp_path) -> None:
    input_csv = tmp_path / "events.csv"
    input_csv.write_text("timestamp,text\n2026-01-01T10:00:00Z,hello\n", encoding="utf-8")

    report = validate_unified_table(input_csv)

    assert report.is_valid is True
    assert report.n_rows == 1


def test_derive_columns_uses_mappers_for_blank_values() -> None:
    source = [
        {"timestamp": "2026-01-01T10:00:00Z", "text": "hi", "actor_id": "", "event_type": None},
    ]
    derived = derive_columns(
        source,
        actor_mapper=lambda row: f"user:{row['text']}",
        event_mapper=lambda _row: "message",
    )
    assert derived[0]["actor_id"] == "user:hi"
    assert derived[0]["event_type"] == "message"


def test_coerce_unified_table_rejects_inconsistent_columnar_input() -> None:
    with pytest.raises(ValueError, match="equally sized"):
        coerce_unified_table({"timestamp": ["2026-01-01"], "text": ["a", "b"]})
