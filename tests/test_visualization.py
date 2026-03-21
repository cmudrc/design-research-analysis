from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest

from design_research_analysis import (
    plot_convergence_curve,
    plot_design_process_timeline,
    plot_idea_trajectory,
)
from design_research_analysis.visualization import _timeline_sort_key


def test_plot_design_process_timeline_returns_figure_and_axis() -> None:
    rows = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "session_id": "session-a",
            "actor_id": "alice",
            "event_type": "propose",
        },
        {
            "timestamp": "2026-01-01T10:01:00Z",
            "session_id": "session-a",
            "actor_id": "bob",
            "event_type": "evaluate",
        },
    ]

    fig, ax = plot_design_process_timeline(rows)

    assert fig is ax.figure
    assert ax.get_title() == "Design Process Timeline"
    plt.close(fig)


def test_plot_design_process_timeline_requires_explicit_session_for_multiple_sessions() -> None:
    rows = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "session_id": "session-a",
            "actor_id": "alice",
            "event_type": "propose",
        },
        {
            "timestamp": "2026-01-01T10:01:00Z",
            "session_id": "session-b",
            "actor_id": "bob",
            "event_type": "evaluate",
        },
    ]

    with pytest.raises(ValueError, match="single session"):
        plot_design_process_timeline(rows)


def test_plot_design_process_timeline_supports_session_filter_and_event_order_axis() -> None:
    rows = [
        {
            "timestamp": "",
            "session_id": "session-a",
            "actor_id": "alice",
            "event_type": "propose",
        },
        {
            "timestamp": "",
            "session_id": "session-b",
            "actor_id": "bob",
            "event_type": "evaluate",
        },
    ]

    fig, ax = plt.subplots()
    returned_fig, returned_ax = plot_design_process_timeline(rows, session_id="session-b", ax=ax)

    assert returned_fig is fig
    assert returned_ax is ax
    assert ax.get_xlabel() == "Event Order"
    plt.close(fig)


def test_plot_design_process_timeline_errors_when_session_filter_is_empty() -> None:
    rows = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "session_id": "session-a",
            "actor_id": "alice",
            "event_type": "propose",
        }
    ]

    with pytest.raises(ValueError, match="No events available"):
        plot_design_process_timeline(rows, session_id="missing")


def test_plot_idea_trajectory_returns_figure_and_axis() -> None:
    fig, ax = plot_idea_trajectory(
        np.asarray([[0.0, 0.0], [1.0, 0.5], [1.5, 1.0]]),
        timestamps=[0, 1, 2],
        groups=["session-a", "session-a", "session-a"],
    )

    assert fig is ax.figure
    assert ax.get_title() == "Idea Trajectory"
    plt.close(fig)


def test_plot_idea_trajectory_rejects_non_2d_projection() -> None:
    with pytest.raises(ValueError, match="requires a 2D projection"):
        plot_idea_trajectory(np.asarray([[0.0, 0.0, 0.0], [1.0, 0.5, 0.2]]))


def test_plot_convergence_curve_accepts_series_and_group_mapping() -> None:
    fig_single, ax_single = plot_convergence_curve([0.8, 0.5, 0.2])
    assert fig_single is ax_single.figure
    plt.close(fig_single)

    fig_grouped, ax_grouped = plot_convergence_curve(
        {
            "session-a": [0.8, 0.5, 0.2],
            "session-b": [0.4, 0.3, 0.3],
        }
    )
    assert fig_grouped is ax_grouped.figure
    plt.close(fig_grouped)


def test_plot_convergence_curve_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="mapping must not be empty"):
        plot_convergence_curve({})
    with pytest.raises(ValueError, match="must not be empty"):
        plot_convergence_curve([])


def test_timeline_sort_key_covers_supported_value_types() -> None:
    assert _timeline_sort_key(None, index=0) == (2, "", 0)
    assert _timeline_sort_key(datetime(2026, 1, 1, 10, 0, 0), index=1)[0] == 0
    assert _timeline_sort_key(np.int64(2), index=2)[1] == 2.0
    assert _timeline_sort_key("2026-01-01T10:00:00Z", index=3)[0] == 0
    assert _timeline_sort_key("phase-a", index=3) == (1, "phase-a", 3)
    assert _timeline_sort_key(object(), index=4)[0] == 1
