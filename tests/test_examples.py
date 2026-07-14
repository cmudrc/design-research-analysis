"""Execution tests for examples supported by the base analysis dependencies."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_EXAMPLES = (
    "basic_usage.py",
    "condition_pair_significance.py",
    "embedding_maps_trajectories.py",
    "experiment_artifacts_handoff.py",
    "idea_space_metrics.py",
    "language_custom_embedder.py",
    "sequence_from_table.py",
    "stats_interrater_reliability.py",
    "stats_regression.py",
    "unified_table_validation.py",
)


@pytest.mark.parametrize("example_name", BASE_EXAMPLES)
def test_base_example_runs(
    example_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each base-compatible example should execute without network access."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [example_name])
    runpy.run_path(str(REPO_ROOT / "examples" / example_name), run_name="__main__")
