"""Top-level visualization helpers for design-process analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .dimred import ProjectionResult, _coerce_feature_matrix, compute_idea_space_trajectory
from .table import coerce_unified_table


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _timeline_sort_key(value: Any, *, index: int) -> tuple[int, float | str, int]:
    if _is_blank(value):
        return (2, "", index)
    if isinstance(value, datetime):
        return (0, float(value.timestamp()), index)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)):
        return (0, float(value), index)
    if isinstance(value, str):
        stripped = value.strip()
        normalized = stripped.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return (1, stripped, index)
        return (0, float(parsed.timestamp()), index)
    return (1, str(value), index)


def _resolve_axis(ax: Axes | None, *, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    if ax is None:
        return plt.subplots(figsize=figsize)
    return cast(Figure, ax.figure), ax


def plot_design_process_timeline(
    events: Sequence[Mapping[str, Any]],
    *,
    session_id: str | None = None,
    session_column: str = "session_id",
    actor_column: str = "actor_id",
    event_column: str = "event_type",
    timestamp_column: str = "timestamp",
    ax: Axes | None = None,
    title: str = "Design Process Timeline",
) -> tuple[Figure, Axes]:
    """Plot one session as an actor-by-time event timeline.

    Args:
        events: Unified-table rows to visualize.
        session_id: Explicit session to render when multiple sessions are present.
        session_column: Session identifier column.
        actor_column: Actor identifier column.
        event_column: Event label column.
        timestamp_column: Timestamp column used for ordering.
        ax: Optional Matplotlib axis.
        title: Plot title.

    Returns:
        ``(figure, axis)`` tuple.
    """
    rows = coerce_unified_table(events)
    sessions = {
        str(row.get(session_column)) for row in rows if not _is_blank(row.get(session_column))
    }
    if session_id is None and len(sessions) > 1:
        raise ValueError("Timeline plotting requires a single session or an explicit session_id.")

    if session_id is not None:
        rows = [row for row in rows if str(row.get(session_column)) == session_id]
    if not rows:
        raise ValueError("No events available for timeline plotting.")

    ordered = [
        row
        for _, row in sorted(
            enumerate(rows),
            key=lambda pair: _timeline_sort_key(pair[1].get(timestamp_column), index=pair[0]),
        )
    ]

    actors: list[str] = []
    for row in ordered:
        actor = row.get(actor_column)
        label = "unknown" if _is_blank(actor) else str(actor)
        if label not in actors:
            actors.append(label)
    actor_positions = {actor: idx for idx, actor in enumerate(actors)}

    timestamps = [row.get(timestamp_column) for row in ordered]
    if all(isinstance(timestamp, datetime) for timestamp in timestamps):
        base_time = cast(datetime, timestamps[0])
        x_positions = [
            float((cast(datetime, timestamp) - base_time).total_seconds())
            for timestamp in timestamps
        ]
        x_label = "Elapsed Seconds"
    else:
        x_positions = [float(index) for index in range(len(ordered))]
        x_label = "Event Order"

    y_positions = [
        float(
            actor_positions[
                "unknown" if _is_blank(row.get(actor_column)) else str(row.get(actor_column))
            ]
        )
        for row in ordered
    ]
    event_types = [
        "event" if _is_blank(row.get(event_column)) else str(row.get(event_column))
        for row in ordered
    ]
    unique_event_types = list(dict.fromkeys(event_types))
    palette = plt.get_cmap("tab10")

    fig, resolved_ax = _resolve_axis(ax, figsize=(9, 4.8))
    resolved_ax.plot(x_positions, y_positions, color="#cbd5e1", linewidth=1.6, zorder=1)
    for index, event_type in enumerate(unique_event_types):
        member_positions = [i for i, value in enumerate(event_types) if value == event_type]
        resolved_ax.scatter(
            [x_positions[i] for i in member_positions],
            [y_positions[i] for i in member_positions],
            s=85,
            color=palette(index % 10),
            edgecolors="white",
            linewidths=0.8,
            label=event_type,
            zorder=2,
        )

    resolved_ax.set_yticks(list(actor_positions.values()))
    resolved_ax.set_yticklabels(list(actor_positions))
    resolved_ax.set_xlabel(x_label)
    resolved_ax.set_ylabel("Actor")
    resolved_ax.set_title(title)
    resolved_ax.grid(axis="x", color="#e2e8f0", linewidth=0.8, alpha=0.9)
    resolved_ax.legend(title="Event Type", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return fig, resolved_ax


def plot_idea_trajectory(
    projection: Sequence[Sequence[float]] | np.ndarray | ProjectionResult,
    *,
    groups: Sequence[Any] | None = None,
    timestamps: Sequence[Any] | None = None,
    ax: Axes | None = None,
    title: str = "Idea Trajectory",
) -> tuple[Figure, Axes]:
    """Plot ordered 2D trajectories through idea space.

    Args:
        projection: Two-dimensional point matrix or projection result.
        groups: Optional group labels that split the trajectory into paths.
        timestamps: Optional timestamps used to order points within each group.
        ax: Optional Matplotlib axis.
        title: Plot title.

    Returns:
        ``(figure, axis)`` tuple.
    """
    matrix, _ = _coerce_feature_matrix(projection, name="projection")
    if matrix.shape[1] != 2:
        raise ValueError("Idea trajectory plotting requires a 2D projection.")

    trajectory = compute_idea_space_trajectory(matrix, timestamps=timestamps, groups=groups)
    fig, resolved_ax = _resolve_axis(ax, figsize=(7.2, 5.6))
    palette = plt.get_cmap("tab10")

    for index, (group, payload) in enumerate(trajectory["groups"].items()):
        points = np.asarray(payload["points"], dtype=float)
        color = palette(index % 10)
        resolved_ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=str(group),
        )
        resolved_ax.scatter(
            points[0, 0],
            points[0, 1],
            color=color,
            s=95,
            marker="o",
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )
        resolved_ax.scatter(
            points[-1, 0],
            points[-1, 1],
            color=color,
            s=110,
            marker="X",
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        resolved_ax.annotate(
            "start",
            (points[0, 0], points[0, 1]),
            xytext=(6, 6),
            textcoords="offset points",
            color=color,
            fontsize=8,
        )
        resolved_ax.annotate(
            "end",
            (points[-1, 0], points[-1, 1]),
            xytext=(6, -10),
            textcoords="offset points",
            color=color,
            fontsize=8,
        )

    resolved_ax.set_xlabel("Component 1")
    resolved_ax.set_ylabel("Component 2")
    resolved_ax.set_title(title)
    resolved_ax.grid(color="#e2e8f0", linewidth=0.8, alpha=0.9)
    resolved_ax.legend(title="Group")
    fig.tight_layout()
    return fig, resolved_ax


def plot_convergence_curve(
    metric_series: Sequence[float] | Mapping[str, Sequence[float]],
    *,
    ax: Axes | None = None,
    title: str = "Convergence Curve",
    ylabel: str = "Metric Value",
) -> tuple[Figure, Axes]:
    """Plot one or more stepwise convergence or divergence curves.

    Args:
        metric_series: Either a single numeric series or ``group -> series``.
        ax: Optional Matplotlib axis.
        title: Plot title.
        ylabel: Y-axis label.

    Returns:
        ``(figure, axis)`` tuple.
    """
    if isinstance(metric_series, Mapping):
        series_map = {
            str(group): [float(value) for value in values]
            for group, values in metric_series.items()
        }
        if not series_map:
            raise ValueError("metric_series mapping must not be empty.")
    else:
        values = [float(value) for value in metric_series]
        if not values:
            raise ValueError("metric_series must not be empty.")
        series_map = {"series": values}

    fig, resolved_ax = _resolve_axis(ax, figsize=(7.0, 4.6))
    palette = plt.get_cmap("tab10")
    for index, (group, values) in enumerate(series_map.items()):
        resolved_ax.plot(
            list(range(len(values))),
            values,
            color=palette(index % 10),
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label=group,
        )

    resolved_ax.set_xlabel("Step")
    resolved_ax.set_ylabel(ylabel)
    resolved_ax.set_title(title)
    resolved_ax.grid(color="#e2e8f0", linewidth=0.8, alpha=0.9)
    if len(series_map) > 1:
        resolved_ax.legend(title="Series")
    fig.tight_layout()
    return fig, resolved_ax


__all__ = [
    "plot_convergence_curve",
    "plot_design_process_timeline",
    "plot_idea_trajectory",
]
