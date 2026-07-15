"""Tests for the experiments artifact integration helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from design_research_analysis.integration import (
    build_condition_metric_table_from_artifacts,
    build_event_table_from_artifacts,
    build_run_metric_table_from_artifacts,
    compare_condition_pairs_from_artifacts,
    compare_markov_chains_from_artifacts,
    fit_markov_chains_from_artifacts,
    fit_regression_from_artifacts,
    load_experiment_artifacts,
    validate_experiment_events,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_canonical_artifacts(output_dir: Path, *, schema_version: str = "0.1.0") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps({"schema_version": schema_version, "study_id": "demo-study"}),
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


def _write_factorial_artifacts(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "0.1.0", "study_id": "factorial-study"}),
        encoding="utf-8",
    )
    conditions = [
        {
            "study_id": "factorial-study",
            "condition_id": "cond-1",
            "agent_treatment": "baseline",
            "model_size_b": 7,
            "task_family": "mechanical",
            "design_task": "bracket",
            "admissible": True,
            "constraint_messages": "[]",
            "assignment_meta_json": "{}",
        },
        {
            "study_id": "factorial-study",
            "condition_id": "cond-2",
            "agent_treatment": "baseline",
            "model_size_b": 13,
            "task_family": "thermal",
            "design_task": "heat_sink",
            "admissible": True,
            "constraint_messages": "[]",
            "assignment_meta_json": "{}",
        },
        {
            "study_id": "factorial-study",
            "condition_id": "cond-3",
            "agent_treatment": "planner",
            "model_size_b": 7,
            "task_family": "thermal",
            "design_task": "heat_sink",
            "admissible": True,
            "constraint_messages": "[]",
            "assignment_meta_json": "{}",
        },
        {
            "study_id": "factorial-study",
            "condition_id": "cond-4",
            "agent_treatment": "planner",
            "model_size_b": 13,
            "task_family": "mechanical",
            "design_task": "bracket",
            "admissible": True,
            "constraint_messages": "[]",
            "assignment_meta_json": "{}",
        },
    ]
    _write_csv(output_dir / "conditions.csv", conditions)

    runs = []
    scores = {
        "run-1": 0.40,
        "run-2": 0.55,
        "run-3": 0.80,
        "run-4": 0.95,
    }
    for index, condition in enumerate(conditions, start=1):
        run_id = f"run-{index}"
        runs.append(
            {
                "study_id": "factorial-study",
                "condition_id": condition["condition_id"],
                "run_id": run_id,
                "problem_id": condition["design_task"],
                "problem_family": condition["task_family"],
                "agent_id": condition["agent_treatment"],
                "agent_kind": "llm",
                "pattern_name": "ideation",
                "model_name": f"model-{condition['model_size_b']}b",
                "seed": index,
                "replicate": 1,
                "status": "success",
                "start_time": "2026-01-01T00:00:00Z",
                "end_time": "2026-01-01T00:00:03Z",
                "latency_s": 3.0,
                "input_tokens": 10,
                "output_tokens": 20,
                "cost_usd": 0.0,
                "primary_outcome": scores[run_id],
                "trace_path": "",
                "manifest_path": "manifest.json",
            }
        )
    _write_csv(output_dir / "runs.csv", runs)

    event_rows = []
    for index, run in enumerate(runs, start=1):
        treatment = run["agent_id"]
        middle_event = "simulate" if treatment == "planner" else "revise"
        for step, event_type in enumerate(("inspect", middle_event, "submit")):
            event_rows.append(
                {
                    "timestamp": f"2026-01-01T00:00:{index}{step}Z",
                    "record_id": f"evt-{index}-{step}",
                    "text": event_type,
                    "session_id": run["run_id"],
                    "actor_id": "agent",
                    "event_type": event_type,
                    "meta_json": "{}",
                    "run_id": run["run_id"],
                }
            )
    _write_csv(output_dir / "events.csv", event_rows)

    _write_csv(
        output_dir / "evaluations.csv",
        [
            {
                "run_id": run_id,
                "evaluator_id": "rubric",
                "metric_name": "quality_score",
                "metric_value": score,
                "metric_unit": "unitless",
                "aggregation_level": "run",
                "notes_json": "{}",
            }
            for run_id, score in scores.items()
        ],
    )


@pytest.mark.parametrize("schema_version", ["0.1.0", "0.2.0"])
def test_load_experiment_artifacts_reads_supported_canonical_export(
    tmp_path: Path, schema_version: str
) -> None:
    output_dir = tmp_path / "study-output"
    _write_canonical_artifacts(output_dir, schema_version=schema_version)

    artifacts = load_experiment_artifacts(output_dir)

    assert artifacts["manifest.json"]["schema_version"] == schema_version
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


def test_load_experiment_artifacts_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    output_dir = tmp_path / "study-output"
    _write_canonical_artifacts(output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "9.0.0", "study_id": "demo-study"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Unsupported.*9\.0\.0.*0\.1\.0, 0\.2\.0"):
        load_experiment_artifacts(output_dir)


def test_validate_experiment_events_requires_versioned_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "study-output"
    _write_canonical_artifacts(output_dir)
    (output_dir / "manifest.json").unlink()

    with pytest.raises(ValueError, match=r"manifest\.json"):
        validate_experiment_events(output_dir / "events.csv")


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


def test_build_condition_metric_table_from_artifacts_reads_context_and_scores(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    rows = build_condition_metric_table_from_artifacts(
        output_dir,
        metric="quality_score",
        condition_column="agent_treatment",
    )

    assert {row["condition"] for row in rows} == {"baseline", "planner"}
    assert [row["metric_source"] for row in rows] == ["evaluations"] * 4
    assert sum(row["value"] for row in rows if row["condition"] == "planner") == pytest.approx(1.75)


def test_compare_condition_pairs_from_artifacts_returns_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    report = compare_condition_pairs_from_artifacts(
        output_dir,
        metric="quality_score",
        condition_column="agent_treatment",
        condition_pairs=[("planner", "baseline")],
        alternative="greater",
        seed=17,
    )

    comparison = report.comparisons[0]
    assert report.metric == "quality_score"
    assert comparison.left_condition == "planner"
    assert comparison.mean_difference > 0


def test_build_event_table_from_artifacts_adds_run_and_condition_context(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    rows = build_event_table_from_artifacts(output_dir)

    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["agent_treatment"] == "baseline"
    assert rows[0]["model_size_b"] == "7"
    assert rows[0]["problem_family"] == "mechanical"


def test_fit_markov_chains_from_artifacts_groups_by_condition(tmp_path: Path) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    chains = fit_markov_chains_from_artifacts(
        output_dir,
        condition_column="agent_treatment",
        smoothing=0.5,
    )

    assert sorted(chains) == ["baseline", "planner"]
    assert chains["baseline"].n_sequences == 2
    assert chains["planner"].config["source"] == "experiment_artifacts"


def test_compare_markov_chains_from_artifacts_returns_transition_comparison(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    comparison = compare_markov_chains_from_artifacts(
        output_dir,
        condition_column="agent_treatment",
        left_condition="planner",
        right_condition="baseline",
    )

    assert comparison.metric == "transition_profile"
    assert comparison.details["n_profile_cells"] > 0


def test_build_run_metric_table_from_artifacts_returns_wide_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    rows = build_run_metric_table_from_artifacts(output_dir, metrics="quality_score")

    assert rows[0]["quality_score"] == pytest.approx(0.40)
    assert rows[0]["agent_treatment"] == "baseline"
    assert rows[0]["model_size_b"] == "7"


def test_build_run_metric_table_from_artifacts_reads_run_metric_columns(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    rows = build_run_metric_table_from_artifacts(output_dir, metrics="primary_outcome")

    assert rows[0]["primary_outcome"] == pytest.approx(0.40)


def test_build_run_metric_table_from_artifacts_rejects_blank_metric_names(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    with pytest.raises(ValueError, match="blank"):
        build_run_metric_table_from_artifacts(output_dir, metrics="")
    with pytest.raises(ValueError, match="empty"):
        build_run_metric_table_from_artifacts(output_dir, metrics=())


def test_build_event_table_from_artifacts_rejects_missing_requested_context(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    with pytest.raises(ValueError, match="missing requested columns"):
        build_event_table_from_artifacts(output_dir, condition_columns=("missing_factor",))


def test_fit_markov_chains_from_artifacts_rejects_missing_group_column(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    with pytest.raises(ValueError, match="missing condition column"):
        fit_markov_chains_from_artifacts(output_dir, condition_column="missing_factor")


def test_fit_markov_chains_from_artifacts_requires_two_groups(tmp_path: Path) -> None:
    output_dir = tmp_path / "study-output"
    _write_canonical_artifacts(output_dir)

    with pytest.raises(ValueError, match="At least two condition groups"):
        fit_markov_chains_from_artifacts(output_dir, condition_column="condition_id")


def test_compare_markov_chains_from_artifacts_rejects_unknown_conditions(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    with pytest.raises(ValueError, match="Unknown condition"):
        compare_markov_chains_from_artifacts(
            output_dir,
            condition_column="agent_treatment",
            left_condition="planner",
            right_condition="missing",
        )


def test_fit_regression_from_artifacts_rejects_empty_predictors(tmp_path: Path) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    with pytest.raises(ValueError, match="predictors"):
        fit_regression_from_artifacts(output_dir, outcome="quality_score", predictors=())


def test_fit_regression_from_artifacts_encodes_numeric_and_categorical_predictors(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    result = fit_regression_from_artifacts(
        output_dir,
        outcome="quality_score",
        predictors=("model_size_b", "task_family"),
        categorical_predictors=("task_family",),
    )

    assert result.n_samples == 4
    assert set(result.coefficients) == {"model_size_b", "task_family[thermal]"}


def test_fit_regression_from_artifacts_can_use_evaluation_metric_predictors(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "study-output"
    _write_factorial_artifacts(output_dir)

    result = fit_regression_from_artifacts(
        output_dir,
        outcome="primary_outcome",
        predictors=("quality_score",),
    )

    assert result.coefficients["quality_score"] == pytest.approx(1.0)
