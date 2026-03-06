"""Visualization helpers for Markov-chain and HMM transition dynamics."""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .models import _state_labels, _transition_like_matrix

_SEQ_IMPORT_ERROR = (
    "State-graph visualization requires optional sequence dependencies. "
    "Install with `pip install design-research-analysis[seq]`."
)


def _resolve_transition_and_labels(
    transition: Any,
    state_labels: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    try:
        matrix = np.asarray(_transition_like_matrix(transition), dtype=float)
        default_labels = _state_labels(transition)
    except TypeError:
        matrix = np.asarray(transition, dtype=float)
        default_labels = [f"S{idx}" for idx in range(matrix.shape[0])] if matrix.ndim == 2 else []

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Transition matrix must be square.")

    labels = state_labels or default_labels
    if len(labels) != matrix.shape[0]:
        raise ValueError("state_labels length must match transition matrix size.")
    return matrix, labels


def plot_transition_matrix(
    transition: Any,
    *,
    state_labels: list[str] | None = None,
    ax: Axes | None = None,
    cmap: str = "Blues",
    annotate: bool = True,
    fmt: str = ".2f",
    title: str = "Transition Matrix",
) -> tuple[Figure, Axes]:
    """Plot a transition matrix as a heatmap.

    Args:
        transition: Result object or raw square matrix.
        state_labels: Optional display labels for states.
        ax: Optional Matplotlib axis.
        cmap: Heatmap colormap.
        annotate: Whether to annotate each cell with probabilities.
        fmt: Format string for annotations.
        title: Plot title.

    Returns:
        ``(figure, axis)`` tuple.
    """
    matrix, labels = _resolve_transition_and_labels(transition, state_labels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = cast(Figure, ax.figure)

    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=max(1.0, float(matrix.max())))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(matrix.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_title(title)

    if annotate:
        formatter = "{" + f":{fmt}" + "}"
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = formatter.format(matrix[row, col])
                color = "white" if matrix[row, col] > (matrix.max() * 0.5) else "black"
                ax.text(col, row, value, ha="center", va="center", color=color)

    return fig, ax


def plot_state_graph(
    transition: Any,
    *,
    state_labels: list[str] | None = None,
    threshold: float = 0.0,
    ax: Axes | None = None,
    seed: int = 0,
    title: str = "State Transition Graph",
) -> tuple[Figure, Axes]:
    """Render a directed state-transition graph.

    Args:
        transition: Result object or raw square matrix.
        state_labels: Optional display labels for states.
        threshold: Draw edges with probability strictly above this value.
        ax: Optional Matplotlib axis.
        seed: Random seed passed to layout generation.
        title: Plot title.

    Returns:
        ``(figure, axis)`` tuple.
    """
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")

    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(_SEQ_IMPORT_ERROR) from exc

    matrix, labels = _resolve_transition_and_labels(transition, state_labels)

    graph = nx.DiGraph()
    for idx, label in enumerate(labels):
        graph.add_node(idx, label=label)

    for src in range(matrix.shape[0]):
        for dst in range(matrix.shape[1]):
            weight = float(matrix[src, dst])
            if weight > threshold:
                graph.add_edge(src, dst, weight=weight)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = cast(Figure, ax.figure)

    if graph.number_of_nodes() == 0:
        ax.set_title(title)
        ax.axis("off")
        return fig, ax

    positions = nx.spring_layout(graph, seed=seed)
    nx.draw_networkx_nodes(graph, positions, node_size=1400, node_color="#d0e3ff", ax=ax)
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={idx: labels[idx] for idx in range(len(labels))},
        font_size=9,
        ax=ax,
    )

    edge_widths = [max(0.8, 6.0 * graph[u][v]["weight"]) for u, v in graph.edges]
    nx.draw_networkx_edges(
        graph,
        positions,
        width=edge_widths,
        edge_color="#4f5d75",
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.12",
        ax=ax,
    )

    edge_labels = {(u, v): f"{graph[u][v]['weight']:.2f}" for u, v in graph.edges}
    if edge_labels:
        nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_size=8, ax=ax)

    ax.set_title(title)
    ax.axis("off")
    return fig, ax


__all__ = ["plot_state_graph", "plot_transition_matrix"]
