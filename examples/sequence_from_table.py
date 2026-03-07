"""Fit a Markov chain from event-coded lab session rows.

## Introduction
Convert a short event log into a transition model that describes the team's
design-process flow.

## Technical Implementation
1. Construct event rows with timestamps, session IDs, and event labels.
2. Fit a first-order Markov chain with additive smoothing.
3. Print state order and transition probabilities.

## Expected Results
Prints a tuple state list and a dense transition matrix suitable for downstream
visualization or comparison across sessions.

## References
- docs/workflows.rst
"""

from __future__ import annotations

import design_research_analysis as dran


def main() -> None:
    """Fit and print a transition matrix."""
    rows = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "session_id": "s1",
            "event_type": "propose",
        },
        {
            "timestamp": "2026-01-01T10:01:00Z",
            "session_id": "s1",
            "event_type": "evaluate",
        },
        {
            "timestamp": "2026-01-01T10:02:00Z",
            "session_id": "s1",
            "event_type": "refine",
        },
        {
            "timestamp": "2026-01-01T10:03:00Z",
            "session_id": "s1",
            "event_type": "evaluate",
        },
    ]
    result = dran.fit_markov_chain_from_table(rows, order=1, smoothing=1.0)
    print(result.states)
    print(result.transition_matrix)


if __name__ == "__main__":
    main()
