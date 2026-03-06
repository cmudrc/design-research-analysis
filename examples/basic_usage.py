"""End-to-end example using the unified-table workflow."""

from __future__ import annotations

from design_research_analysis import (
    compute_language_convergence,
    fit_markov_chain_from_table,
    validate_unified_table,
)


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

    report = validate_unified_table(rows)
    if not report.is_valid:
        raise RuntimeError(f"Invalid table: {report.errors}")

    markov = fit_markov_chain_from_table(rows, order=1, smoothing=1.0)
    print("Markov states:", markov.states)
    print("Transition matrix:")
    print(markov.transition_matrix)

    custom_vectors = {
        "initial concept sketch": [3.0, 0.0],
        "review constraints and compare options": [1.5, 0.0],
        "refined concept around shared constraints": [0.0, 0.0],
    }
    convergence = compute_language_convergence(
        rows,
        window_size=1,
        embedder=lambda texts: [custom_vectors[text] for text in texts],
    )
    print("Language direction:", convergence.direction_by_group)


if __name__ == "__main__":
    main()
