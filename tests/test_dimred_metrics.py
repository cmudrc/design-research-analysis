from __future__ import annotations

import numpy as np
import pytest

from design_research_analysis.dimred import (
    ProjectionResult,
    compute_design_space_coverage,
    compute_divergence_convergence,
    compute_idea_space_trajectory,
)


def test_compute_design_space_coverage_accepts_projection_results() -> None:
    projection = ProjectionResult(
        projection=np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        method="pca",
        config={"n_components": 2},
    )

    coverage = compute_design_space_coverage(projection)

    assert coverage["n_points"] == 4
    assert coverage["pairwise_spread"]["n_pairs"] == 6
    assert coverage["convex_hull"]["supported"] is True
    assert coverage["convex_hull"]["area"] == pytest.approx(1.0)
    assert coverage["config"]["input_source"] == "projection_result"


def test_compute_design_space_coverage_handles_degenerate_cases() -> None:
    with pytest.raises(ValueError, match="2D matrix"):
        compute_design_space_coverage(np.asarray([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="at least one row"):
        compute_design_space_coverage(np.empty((0, 2)))

    three_dimensional = compute_design_space_coverage(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )
    )
    assert three_dimensional["convex_hull"]["area"] is None
    assert "only available for 2D inputs" in three_dimensional["warnings"][0]

    collinear = compute_design_space_coverage(
        np.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )
    )
    assert collinear["convex_hull"]["supported"] is False
    assert collinear["convex_hull"]["area"] is None


def test_compute_idea_space_trajectory_sorts_within_groups_by_timestamp() -> None:
    points = np.asarray(
        [
            [10.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
        ]
    )

    trajectory = compute_idea_space_trajectory(
        points,
        timestamps=[
            "2026-01-01T00:00:10Z",
            "2026-01-01T00:00:01Z",
            "2026-01-01T00:00:02Z",
            "2026-01-01T00:00:03Z",
        ],
        groups=["b", "a", "a", "a"],
    )

    assert trajectory["n_groups"] == 2
    assert trajectory["groups"]["a"]["ordered_indices"] == [1, 2, 3]
    assert trajectory["groups"]["a"]["step_sizes"] == [2.0, 1.0]
    assert trajectory["groups"]["a"]["path_length"] == pytest.approx(3.0)
    assert trajectory["groups"]["b"]["ordered_indices"] == [0]


def test_compute_idea_space_trajectory_defaults_to_global_group() -> None:
    trajectory = compute_idea_space_trajectory(np.asarray([[0.0, 0.0], [1.0, 0.0]]))

    assert trajectory["n_groups"] == 1
    assert list(trajectory["groups"]) == ["__all__"]
    assert trajectory["groups"]["__all__"]["path_length"] == pytest.approx(1.0)


def test_compute_divergence_convergence_labels_rolling_phases() -> None:
    trajectory = {
        "groups": {
            "session-a": {
                "centroid_distances": [0.1, 0.4, 0.7, 0.5, 0.2],
            }
        }
    }

    summary = compute_divergence_convergence(trajectory, window=2)

    phases = [marker["phase"] for marker in summary["groups"]["session-a"]["phase_markers"]]
    assert phases == ["stable", "diverging", "diverging", "converging"]
    assert summary["groups"]["session-a"]["dominant_direction"] == "diverging"
    assert summary["groups"]["session-a"]["convergence_rate"] == pytest.approx(1 / 3)
