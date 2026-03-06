"""Compute language convergence with a custom deterministic embedder."""

from __future__ import annotations

from design_research_analysis import compute_language_convergence


def main() -> None:
    """Run language convergence analysis without optional embedding deps."""
    rows = [
        {"timestamp": "2026-01-01T10:00:00Z", "session_id": "s1", "text": "far"},
        {"timestamp": "2026-01-01T10:01:00Z", "session_id": "s1", "text": "mid"},
        {"timestamp": "2026-01-01T10:02:00Z", "session_id": "s1", "text": "target"},
    ]

    lookup = {
        "far": [0.0, 1.0],
        "mid": [0.6, 0.6],
        "target": [1.0, 0.0],
    }
    result = compute_language_convergence(
        rows,
        window_size=1,
        embedder=lambda texts: [lookup[text] for text in texts],
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()
