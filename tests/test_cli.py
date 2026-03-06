from __future__ import annotations

import csv
import json
from pathlib import Path

from design_research_analysis.cli import main


def _write_fixture_csv(path: Path) -> None:
    rows = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "session_id": "s1",
            "event_type": "A",
            "text": "good clear result",
            "record_id": "r1",
            "value": "1.0",
            "group": "g1",
            "x1": "1.0",
            "x2": "0.0",
            "y": "2.0",
        },
        {
            "timestamp": "2026-01-01T10:00:01Z",
            "session_id": "s1",
            "event_type": "B",
            "text": "bad unclear problem",
            "record_id": "r2",
            "value": "2.0",
            "group": "g1",
            "x1": "2.0",
            "x2": "1.0",
            "y": "4.0",
        },
        {
            "timestamp": "2026-01-01T10:00:02Z",
            "session_id": "s2",
            "event_type": "A",
            "text": "great collaborative success",
            "record_id": "r3",
            "value": "5.0",
            "group": "g2",
            "x1": "3.0",
            "x2": "1.0",
            "y": "6.0",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_cli_validate_table_smoke(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    summary_json = tmp_path / "validate.json"
    _write_fixture_csv(input_csv)

    exit_code = main(
        [
            "validate-table",
            "--input",
            str(input_csv),
            "--summary-json",
            str(summary_json),
        ]
    )

    assert exit_code == 0
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["is_valid"] is True


def test_cli_run_sequence_markov_smoke(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    summary_json = tmp_path / "sequence.json"
    _write_fixture_csv(input_csv)

    exit_code = main(
        [
            "run-sequence",
            "--input",
            str(input_csv),
            "--summary-json",
            str(summary_json),
            "--mode",
            "markov",
        ]
    )

    assert exit_code == 0
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["mode"] == "markov"
    assert "transition_matrix" in payload["result"]


def test_cli_run_stats_regression_smoke(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    summary_json = tmp_path / "stats.json"
    _write_fixture_csv(input_csv)

    exit_code = main(
        [
            "run-stats",
            "--input",
            str(input_csv),
            "--summary-json",
            str(summary_json),
            "--mode",
            "regression",
            "--x-columns",
            "x1,x2",
            "--y-column",
            "y",
        ]
    )

    assert exit_code == 0
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["mode"] == "regression"
    assert "coefficients" in payload["result"]
