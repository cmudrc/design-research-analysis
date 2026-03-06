"""Fit a Markov chain directly from unified-table rows."""

from __future__ import annotations

from design_research_analysis import fit_markov_chain_from_table


def main() -> None:
    """Fit and print a transition matrix."""
    rows = [
        {"timestamp": "2026-01-01T10:00:00Z", "session_id": "s1", "event_type": "propose"},
        {"timestamp": "2026-01-01T10:01:00Z", "session_id": "s1", "event_type": "evaluate"},
        {"timestamp": "2026-01-01T10:02:00Z", "session_id": "s1", "event_type": "refine"},
        {"timestamp": "2026-01-01T10:03:00Z", "session_id": "s1", "event_type": "evaluate"},
    ]
    result = fit_markov_chain_from_table(rows, order=1, smoothing=1.0)
    print(result.states)
    print(result.transition_matrix)


if __name__ == "__main__":
    main()
