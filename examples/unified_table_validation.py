"""Validate and normalize a loose unified-table input."""

from __future__ import annotations

from design_research_analysis import derive_columns, validate_unified_table


def main() -> None:
    """Run a simple table validation workflow."""
    rows = [
        {"timestamp": "2026-01-01T10:00:00Z", "text": "hello", "speaker": "alice"},
        {"timestamp": "2026-01-01T10:00:01Z", "text": "world", "speaker": "bob"},
    ]

    rows = derive_columns(
        rows,
        actor_mapper=lambda row: row["speaker"],
        event_mapper=lambda _row: "utterance",
    )
    report = validate_unified_table(rows)
    print(report.to_dict())


if __name__ == "__main__":
    main()
