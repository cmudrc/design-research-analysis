"""Analyze canonical experiment artifacts without manual table joins.

## Introduction
Start from a study-output directory shaped like a
``design-research-experiments`` export and run standard analysis workflows
without manually joining ``runs.csv``, ``conditions.csv``, ``events.csv``, or
``evaluations.csv``.

## Technical Implementation
1. Write a tiny deterministic artifact bundle that stands in for an exported
   experiment.
2. Validate the bundle through top-level artifact helpers.
3. Run condition comparisons, Markov-chain comparisons, and regression directly
   from the artifact directory.

## Expected Results
Prints validation status, derived table sizes, condition-comparison count,
Markov-chain labels, transition-comparison estimate, and regression coefficients.

## References
- docs/experiments_handoff.rst
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import design_research_analysis as dran


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_artifacts(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "0.1.0", "study_id": "artifact-demo"}),
        encoding="utf-8",
    )
    conditions = [
        {
            "study_id": "artifact-demo",
            "condition_id": "cond-baseline-small",
            "agent_treatment": "baseline",
            "model_size_b": 7,
            "task_family": "mechanical",
            "admissible": True,
            "constraint_messages": "[]",
            "assignment_meta_json": "{}",
        },
        {
            "study_id": "artifact-demo",
            "condition_id": "cond-baseline-large",
            "agent_treatment": "baseline",
            "model_size_b": 13,
            "task_family": "thermal",
            "admissible": True,
            "constraint_messages": "[]",
            "assignment_meta_json": "{}",
        },
        {
            "study_id": "artifact-demo",
            "condition_id": "cond-planner-small",
            "agent_treatment": "planner",
            "model_size_b": 7,
            "task_family": "thermal",
            "admissible": True,
            "constraint_messages": "[]",
            "assignment_meta_json": "{}",
        },
        {
            "study_id": "artifact-demo",
            "condition_id": "cond-planner-large",
            "agent_treatment": "planner",
            "model_size_b": 13,
            "task_family": "mechanical",
            "admissible": True,
            "constraint_messages": "[]",
            "assignment_meta_json": "{}",
        },
    ]
    _write_csv(output_dir / "conditions.csv", conditions)

    scores = [0.42, 0.55, 0.78, 0.93]
    runs: list[dict[str, object]] = []
    for index, condition in enumerate(conditions, start=1):
        runs.append(
            {
                "study_id": "artifact-demo",
                "condition_id": condition["condition_id"],
                "run_id": f"run-{index}",
                "problem_id": f"problem-{condition['task_family']}",
                "problem_family": condition["task_family"],
                "agent_id": condition["agent_treatment"],
                "agent_kind": "scripted",
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
                "primary_outcome": scores[index - 1],
                "trace_path": "",
                "manifest_path": "manifest.json",
            }
        )
    _write_csv(output_dir / "runs.csv", runs)

    events: list[dict[str, object]] = []
    for index, run in enumerate(runs, start=1):
        middle_event = "simulate" if run["agent_id"] == "planner" else "revise"
        for step, event_type in enumerate(("inspect", middle_event, "submit")):
            events.append(
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
    _write_csv(output_dir / "events.csv", events)

    _write_csv(
        output_dir / "evaluations.csv",
        [
            {
                "run_id": f"run-{index}",
                "evaluator_id": "rubric",
                "metric_name": "quality_score",
                "metric_value": score,
                "metric_unit": "unitless",
                "aggregation_level": "run",
                "notes_json": "{}",
            }
            for index, score in enumerate(scores, start=1)
        ],
    )


def main() -> None:
    """Run artifact-first analysis workflows over one tiny export."""
    print("Integration module:", dran.integration.__name__)
    with TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "study-output"
        _write_artifacts(output_dir)

        artifacts = dran.load_experiment_artifacts(output_dir)
        validation = dran.validate_experiment_events(output_dir / "events.csv")
        joined_events = dran.build_event_table_from_artifacts(output_dir)
        metric_rows = dran.build_condition_metric_table_from_artifacts(
            output_dir,
            metric="quality_score",
            condition_column="agent_treatment",
        )
        report = dran.compare_condition_pairs_from_artifacts(
            output_dir,
            metric="quality_score",
            condition_column="agent_treatment",
            condition_pairs=[("planner", "baseline")],
            alternative="greater",
            seed=17,
        )
        chains = dran.fit_markov_chains_from_artifacts(
            output_dir,
            condition_column="agent_treatment",
        )
        chain_delta = dran.compare_markov_chains_from_artifacts(
            output_dir,
            condition_column="agent_treatment",
            left_condition="planner",
            right_condition="baseline",
        )
        run_metrics = dran.build_run_metric_table_from_artifacts(
            output_dir,
            metrics="quality_score",
        )
        regression = dran.fit_regression_from_artifacts(
            output_dir,
            outcome="quality_score",
            predictors=("model_size_b", "task_family"),
            categorical_predictors=("task_family",),
        )

        print("Artifact tables:", ", ".join(sorted(artifacts)))
        print("Events valid:", validation.is_valid, f"rows={validation.n_rows}")
        print("Joined event rows:", len(joined_events))
        print("Metric rows:", len(metric_rows), f"run rows={len(run_metrics)}")
        print("Condition comparisons:", len(report.comparisons))
        print("Markov chains:", ", ".join(sorted(chains)))
        print("Transition delta:", f"{chain_delta.estimate:.4f}")
        print("Regression coefficients:", regression.coefficients)


if __name__ == "__main__":
    main()
