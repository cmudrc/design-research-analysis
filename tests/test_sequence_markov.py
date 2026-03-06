from __future__ import annotations

import json

import numpy as np

from design_research_analysis.sequence import fit_markov_chain


def test_fit_markov_chain_rows_are_stochastic() -> None:
    result = fit_markov_chain([["A", "B", "A"], ["A", "A", "B"]], order=1, smoothing=1.0)

    assert result.transition_matrix.shape[0] == result.transition_matrix.shape[1]
    assert np.allclose(result.transition_matrix.sum(axis=1), 1.0)
    assert np.all(result.transition_matrix > 0.0)
    assert np.isclose(result.startprob.sum(), 1.0)


def test_fit_markov_chain_order_two_builds_context_states() -> None:
    result = fit_markov_chain([["x", "y", "z", "y", "z"], ["x", "y", "x", "y"]], order=2)

    assert all(len(state) == 2 for state in result.states)
    assert result.order == 2
    assert np.allclose(result.transition_matrix.sum(axis=1), 1.0)


def test_markov_result_is_json_serializable() -> None:
    result = fit_markov_chain([["S", "NP", "VP", "END"]], smoothing=0.5)
    payload = result.to_dict()

    encoded = json.dumps(payload)
    assert "transition_matrix" in encoded


def test_fit_markov_chain_requires_enough_sequence_length() -> None:
    try:
        fit_markov_chain([["A"], ["B"]], order=1)
        raise AssertionError("Expected ValueError for too-short sequences")
    except ValueError as exc:
        assert "order + 1" in str(exc)
