"""Helpers for consuming canonical experiment exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .table import UnifiedTableValidationReport, coerce_unified_table, validate_unified_table

_ANALYSIS_ARTIFACT_FILES = (
    "manifest.json",
    "conditions.csv",
    "runs.csv",
    "events.csv",
    "evaluations.csv",
)


def load_experiment_artifacts(path: str | Path) -> dict[str, Any]:
    """Load the canonical analysis-facing experiment artifacts.

    Args:
        path: Study output directory or the canonical ``events.csv`` path inside it.

    Returns:
        Mapping keyed by canonical artifact filename.

    Raises:
        ValueError: If ``path`` does not resolve to a canonical artifact directory.
    """
    output_dir = _resolve_output_dir(path)
    missing = [
        artifact_name
        for artifact_name in _ANALYSIS_ARTIFACT_FILES
        if not (output_dir / artifact_name).exists()
    ]
    if missing:
        raise ValueError("Missing canonical experiment artifacts: " + ", ".join(missing) + ".")

    return {
        "manifest.json": _read_json(output_dir / "manifest.json"),
        "conditions.csv": _read_csv(output_dir / "conditions.csv"),
        "runs.csv": _read_csv(output_dir / "runs.csv"),
        "events.csv": _read_csv(output_dir / "events.csv"),
        "evaluations.csv": _read_csv(output_dir / "evaluations.csv"),
    }


def validate_experiment_events(path: str | Path) -> UnifiedTableValidationReport:
    """Validate canonical ``events.csv`` output from design-research-experiments.

    Args:
        path: Study output directory or the canonical ``events.csv`` path inside it.

    Returns:
        Unified-table validation report for the exported event rows.

    Raises:
        ValueError: If ``path`` does not resolve to a canonical ``events.csv`` artifact.
    """
    events_path = _resolve_events_path(path)
    rows = _read_csv(events_path)
    return validate_unified_table(coerce_unified_table(rows))


def _resolve_output_dir(path: str | Path) -> Path:
    """Resolve one study output directory from a directory or events path."""
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        return candidate
    if candidate.is_file() and candidate.name == "events.csv":
        return candidate.parent
    raise ValueError(
        "Expected a study output directory or the canonical 'events.csv' artifact path."
    )


def _resolve_events_path(path: str | Path) -> Path:
    """Resolve the canonical ``events.csv`` path from a directory or file input."""
    candidate = Path(path).expanduser()
    events_path = candidate / "events.csv" if candidate.is_dir() else candidate

    if not events_path.is_file() or events_path.name != "events.csv":
        raise ValueError(
            "Expected a study output directory or the canonical 'events.csv' artifact path."
        )
    return events_path


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read one CSV artifact into row dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON artifact into a dictionary."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object payload in '{path.name}'.")
    return payload


__all__ = ["load_experiment_artifacts", "validate_experiment_events"]
