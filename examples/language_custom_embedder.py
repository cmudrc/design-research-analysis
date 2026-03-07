"""Compute language convergence for a lab session using a deterministic embedder.

## Introduction
Analyze one session's discourse shift without external model dependencies by
supplying a local deterministic embedding lookup.

## Technical Implementation
1. Create three timestamped utterances in one session.
2. Map each utterance to a fixed numeric vector with ``embedder=...``.
3. Compute convergence trajectories and slope-based direction labels.

## Expected Results
Prints a serialized convergence result containing trajectories, per-group slopes,
and direction labels for ``team-b``.

## References
- docs/analysis_recipes.rst
"""

from __future__ import annotations

import design_research_analysis as dran


def main() -> None:
    """Run language convergence analysis without optional embedding deps."""
    rows = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "session_id": "team-b",
            "text": "broad divergent framing",
        },
        {
            "timestamp": "2026-01-01T10:01:00Z",
            "session_id": "team-b",
            "text": "constraint grounded framing",
        },
        {
            "timestamp": "2026-01-01T10:02:00Z",
            "session_id": "team-b",
            "text": "shared actionable framing",
        },
    ]

    lookup = {
        "broad divergent framing": [0.0, 1.0],
        "constraint grounded framing": [0.6, 0.6],
        "shared actionable framing": [1.0, 0.0],
    }
    result = dran.compute_language_convergence(
        rows,
        window_size=1,
        embedder=lambda texts: [lookup[text] for text in texts],
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()
