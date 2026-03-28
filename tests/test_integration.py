"""Tests for the experiments artifact integration helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from design_research_analysis.integration import (
    load_experiment_artifacts,
    validate_experiment_events,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_canonical_artifacts(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "0.1.0", "study_id": "demo-study"}),
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "conditions.csv",
        [
            {
                "study_id": "demo-study",
                "condition_id": "cond-1",
                "admissible": True,
                "constraint_messages": "[]",
                "assignment_meta_json": "{}",
            }
        ],
    )
    _write_csv(
        output_dir / "runs.csv",
        [
            {
                "study_id": "demo-study",
                "condition_id": "cond-1",
                "run_id": "run-1",
                "problem_id": "problem-1",
                "problem_family": "decision",
                "agent_id": "agent-1",
                "agent_kind": "baseline",
                "pattern_name": "workflow",
                "model_name": "test-model",
                "seed": 7,
                "replicate": 1,
                "status": "success",
                "start_time": "2026-01-01T00:00:00Z",
                "end_time": "2026-01-01T00:00:01Z",
                "latency_s": 1.0,
                "input_tokens": 1,
                "output_tokens": 2,
                "cost_usd": 0.0,
                "primary_outcome": 1.0,
                "trace_path": "",
                "manifest_path": "manifest.json",
            }
        ],
    )
    _write_csv(
        output_dir / "events.csv",
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "record_id": "evt-1",
                "text": "hello",
                "session_id": "run-1",
                "actor_id": "agent",
                "event_type": "assistant_output",
                "meta_json": "{}",
                "run_id": "run-1",
            }
        ],
    )
    _write_csv(
        output_dir / "evaluations.csv",
        [
            {
                "run_id": "run-1",
                "evaluator_id": "problem_evaluator",
                "metric_name": "score",
                "metric_value": 1.0,
                "metric_unit": "unitless",
                "aggregation_level": "run",
                "notes_json": "{}",
            }
        ],
    )


def test_load_experiment_artifacts_reads_canonical_export_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "study-output"
    _write_canonical_artifacts(output_dir)

    artifacts = load_experiment_artifacts(output_dir)

    assert artifacts["manifest.json"]["study_id"] == "demo-study"
    assert artifacts["runs.csv"][0]["run_id"] == "run-1"
    assert artifacts["events.csv"][0]["event_type"] == "assistant_output"


def test_load_experiment_artifacts_accepts_events_csv_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "study-output"
    _write_canonical_artifacts(output_dir)

    artifacts = load_experiment_artifacts(output_dir / "events.csv")

    assert artifacts["conditions.csv"][0]["condition_id"] == "cond-1"


def test_load_experiment_artifacts_requires_canonical_siblings(tmp_path: Path) -> None:
    output_dir = tmp_path / "study-output"
    _write_canonical_artifacts(output_dir)
    (output_dir / "runs.csv").unlink()

    with pytest.raises(ValueError, match=r"runs\.csv"):
        load_experiment_artifacts(output_dir)


def test_load_experiment_artifacts_rejects_noncanonical_input_path(tmp_path: Path) -> None:
    bad_path = tmp_path / "study-output" / "notes.csv"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("not canonical", encoding="utf-8")

    with pytest.raises(ValueError, match=r"events\.csv"):
        load_experiment_artifacts(bad_path)


def test_load_experiment_artifacts_requires_json_object_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "study-output"
    _write_canonical_artifacts(output_dir)
    (output_dir / "manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match=r"manifest\.json"):
        load_experiment_artifacts(output_dir)


def test_validate_experiment_events_uses_unified_table_validation(tmp_path: Path) -> None:
    output_dir = tmp_path / "study-output"
    _write_canonical_artifacts(output_dir)

    report = validate_experiment_events(output_dir / "events.csv")

    assert report.is_valid is True
    assert report.n_rows == 1


def test_validate_experiment_events_requires_canonical_events_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "study-output"
    _write_canonical_artifacts(output_dir)

    with pytest.raises(ValueError, match=r"events\.csv"):
        validate_experiment_events(output_dir / "runs.csv")
