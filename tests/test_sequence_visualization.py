from __future__ import annotations

import builtins
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pytest

from design_research_analysis.sequence import fit_markov_chain, plot_state_graph, plot_transition_matrix


def _networkx_available() -> bool:
    return importlib.util.find_spec("networkx") is not None


def test_plot_transition_matrix_returns_figure_and_axis() -> None:
    result = fit_markov_chain([["A", "B", "A", "B"], ["A", "A", "B", "A"]], smoothing=1.0)

    fig, ax = plot_transition_matrix(result, annotate=False)

    assert fig is ax.figure
    assert ax.get_title() == "Transition Matrix"
    plt.close(fig)


def test_plot_transition_matrix_accepts_raw_matrix() -> None:
    fig, ax = plot_transition_matrix(np.array([[0.8, 0.2], [0.3, 0.7]]), annotate=False)

    assert fig is ax.figure
    plt.close(fig)


@pytest.mark.skipif(not _networkx_available(), reason="networkx is unavailable")
def test_plot_state_graph_returns_figure_and_axis() -> None:
    result = fit_markov_chain([["S", "NP", "VP", "END"], ["S", "ADV", "VP", "END"]])

    fig, ax = plot_state_graph(result, threshold=0.1)

    assert fig is ax.figure
    assert ax.get_title() == "State Transition Graph"
    plt.close(fig)


def test_plot_state_graph_surfaces_optional_dependency_error(monkeypatch) -> None:
    original_import = builtins.__import__

    def _mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "networkx":
            raise ImportError("simulated missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)

    result = fit_markov_chain([["S", "NP", "VP", "END"]])
    with pytest.raises(ImportError, match="pip install design-research-analysis\\[seq\\]"):
        plot_state_graph(result)
