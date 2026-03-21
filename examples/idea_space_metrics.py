"""Idea-space metrics and visualization example.

## Introduction
Build a compact idea-space analysis from deterministic vectors, then render the
timeline, trajectory, and convergence views that support interpretation.

## Technical Implementation
1. Construct a single-session unified table plus a deterministic numeric space.
2. Compute projection-space coverage, trajectory summaries, and divergence markers.
3. Render three plots and save them to a stable output directory.

## Expected Results
Prints the convex-hull support flag, the dominant trajectory direction, and the
directory containing the generated PNG figures.

## References
- docs/workflows_dimred.rst
- docs/workflows_sequence.rst
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import design_research_analysis as dran


def parse_args() -> argparse.Namespace:
    """Parse optional example arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/examples/idea_space_metrics",
        help="Directory where PNG files should be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a compact idea-space metrics and visualization workflow."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "session_id": "session-a",
            "actor_id": "alice",
            "event_type": "propose",
            "text": "initial sketch",
        },
        {
            "timestamp": "2026-01-01T10:01:00Z",
            "session_id": "session-a",
            "actor_id": "bob",
            "event_type": "evaluate",
            "text": "compare alternatives",
        },
        {
            "timestamp": "2026-01-01T10:02:00Z",
            "session_id": "session-a",
            "actor_id": "alice",
            "event_type": "refine",
            "text": "merge promising features",
        },
        {
            "timestamp": "2026-01-01T10:03:00Z",
            "session_id": "session-a",
            "actor_id": "bob",
            "event_type": "prototype",
            "text": "externalize the concept",
        },
        {
            "timestamp": "2026-01-01T10:04:00Z",
            "session_id": "session-a",
            "actor_id": "alice",
            "event_type": "evaluate",
            "text": "assess feasibility",
        },
    ]
    vectors = np.asarray(
        [
            [0.0, 0.0],
            [0.6, 0.3],
            [1.3, 1.2],
            [1.8, 1.9],
            [1.2, 1.4],
        ],
        dtype=float,
    )
    timestamps = [row["timestamp"] for row in rows]
    groups = [row["session_id"] for row in rows]

    coverage = dran.compute_design_space_coverage(vectors)
    trajectory = dran.compute_idea_space_trajectory(vectors, timestamps=timestamps, groups=groups)
    dynamics = dran.compute_divergence_convergence(trajectory, window=2)

    timeline_fig, _ = dran.plot_design_process_timeline(rows, session_id="session-a")
    timeline_fig.savefig(output_dir / "design_process_timeline.png", dpi=160, bbox_inches="tight")
    plt.close(timeline_fig)

    trajectory_fig, _ = dran.plot_idea_trajectory(vectors, timestamps=timestamps, groups=groups)
    trajectory_fig.savefig(output_dir / "idea_trajectory.png", dpi=160, bbox_inches="tight")
    plt.close(trajectory_fig)

    session_series = trajectory["groups"]["session-a"]["centroid_distances"]
    convergence_fig, _ = dran.plot_convergence_curve(session_series, ylabel="Centroid Distance")
    convergence_fig.savefig(output_dir / "convergence_curve.png", dpi=160, bbox_inches="tight")
    plt.close(convergence_fig)

    print("Hull supported:", coverage["convex_hull"]["supported"])
    print("Dominant direction:", dynamics["groups"]["session-a"]["dominant_direction"])
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
