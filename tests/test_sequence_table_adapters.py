from __future__ import annotations

import numpy as np
import pytest

from design_research_analysis.sequence import (
    fit_discrete_hmm_from_table,
    fit_markov_chain,
    fit_markov_chain_from_table,
)


def _sample_rows() -> list[dict[str, str]]:
    return [
        {"timestamp": "2026-01-01T10:00:00Z", "session_id": "s1", "event_type": "A"},
        {"timestamp": "2026-01-01T10:00:01Z", "session_id": "s1", "event_type": "B"},
        {"timestamp": "2026-01-01T10:00:02Z", "session_id": "s1", "event_type": "A"},
        {"timestamp": "2026-01-01T10:00:03Z", "session_id": "s2", "event_type": "A"},
        {"timestamp": "2026-01-01T10:00:04Z", "session_id": "s2", "event_type": "B"},
        {"timestamp": "2026-01-01T10:00:05Z", "session_id": "s2", "event_type": "B"},
    ]


def test_fit_markov_chain_from_table_matches_direct_fit() -> None:
    rows = _sample_rows()

    table_result = fit_markov_chain_from_table(rows, order=1, smoothing=1.0)
    direct_result = fit_markov_chain([["A", "B", "A"], ["A", "B", "B"]], order=1, smoothing=1.0)

    assert table_result.states == direct_result.states
    assert np.allclose(table_result.transition_matrix, direct_result.transition_matrix)
    assert np.allclose(table_result.startprob, direct_result.startprob)


def test_fit_discrete_hmm_from_table_rejects_missing_event_token() -> None:
    rows = [{"timestamp": "2026-01-01T10:00:00Z", "session_id": "s1", "event_type": ""}]

    with pytest.raises(ValueError, match="missing 'event_type'"):
        fit_discrete_hmm_from_table(rows)
