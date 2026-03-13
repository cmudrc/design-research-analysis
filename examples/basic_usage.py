"""End-to-end example for a single lab design session.

## Introduction
Use a tiny unified table from one team session to run the most common baseline:
schema validation, event-sequence transitions, and language convergence.

## Technical Implementation
1. Validate required and recommended unified-table columns.
2. Fit a first-order Markov chain from event transitions.
3. Compare the fitted Markov chain to a small alternate session trace.
4. Compute semantic convergence using a deterministic custom embedder.

## Expected Results
Prints the ordered Markov states, transition matrix, one model-comparison summary,
and one convergence label for ``team-a``.

## References
- docs/workflows.rst
"""

from __future__ import annotations

import design_research_analysis as dran


def main() -> None:
    """Run a lightweight unified-table analysis workflow."""
    rows = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "session_id": "team-a",
            "actor_id": "alice",
            "event_type": "propose",
            "text": "initial concept sketch",
        },
        {
            "timestamp": "2026-01-01T10:01:00Z",
            "session_id": "team-a",
            "actor_id": "bob",
            "event_type": "evaluate",
            "text": "review constraints and compare options",
        },
        {
            "timestamp": "2026-01-01T10:02:00Z",
            "session_id": "team-a",
            "actor_id": "alice",
            "event_type": "refine",
            "text": "refined concept around shared constraints",
        },
    ]

    report = dran.validate_unified_table(rows)
    if not report.is_valid:
        raise RuntimeError(f"Invalid table: {report.errors}")

    markov = dran.fit_markov_chain_from_table(rows, order=1, smoothing=1.0)
    print("Markov states:", markov.states)
    print("Transition matrix:")
    print(markov.transition_matrix)

    alternate_rows = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "session_id": "team-b",
            "actor_id": "alice",
            "event_type": "propose",
            "text": "initial concept sketch",
        },
        {
            "timestamp": "2026-01-01T10:01:00Z",
            "session_id": "team-b",
            "actor_id": "bob",
            "event_type": "refine",
            "text": "refined concept around shared constraints",
        },
        {
            "timestamp": "2026-01-01T10:02:00Z",
            "session_id": "team-b",
            "actor_id": "alice",
            "event_type": "evaluate",
            "text": "review constraints and compare options",
        },
    ]
    alternate_markov = dran.fit_markov_chain_from_table(alternate_rows, order=1, smoothing=1.0)
    print("Markov difference:", (markov - alternate_markov).to_dict())

    custom_vectors = {
        "initial concept sketch": [3.0, 0.0],
        "review constraints and compare options": [1.5, 0.0],
        "refined concept around shared constraints": [0.0, 0.0],
    }
    convergence = dran.compute_language_convergence(
        rows,
        window_size=1,
        embedder=lambda texts: [custom_vectors[text] for text in texts],
    )
    print("Language direction:", convergence.direction_by_group)


if __name__ == "__main__":
    main()
