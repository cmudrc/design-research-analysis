"""Validate and normalize a loose transcription-style event table.

## Introduction
Show how lightly structured transcript rows can be normalized into the unified
table contract with deterministic mapper functions.

## Technical Implementation
1. Start from rows with ``speaker`` instead of canonical actor fields.
2. Derive ``actor_id`` and ``event_type`` using mapper callbacks.
3. Validate the resulting rows against the unified-table contract.

## Expected Results
Prints a validation report dictionary with required/recommended field checks and
warnings for missing recommended columns.

## References
- docs/unified_table_schema.rst
"""

from __future__ import annotations

import design_research_analysis as dran


def main() -> None:
    """Run a simple table validation workflow."""
    rows = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "text": "we should decompose the mechanism",
            "speaker": "alice",
        },
        {
            "timestamp": "2026-01-01T10:00:01Z",
            "text": "let's test that with a quick mock",
            "speaker": "bob",
        },
    ]

    rows = dran.derive_columns(
        rows,
        actor_mapper=lambda row: row["speaker"],
        event_mapper=lambda _row: "utterance",
    )
    report = dran.validate_unified_table(rows)
    print(report.to_dict())


if __name__ == "__main__":
    main()
