from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from design_research_analysis.embedding_maps import (
    EmbeddingMapResult,
    EmbeddingResult,
    _json_timestamp,
    _timestamp_sort_key,
    build_embedding_map,
    compute_design_space_coverage,
    compute_divergence_convergence,
    compute_idea_space_trajectory,
    embed_records,
)


def test_compute_design_space_coverage_accepts_embedding_map_results() -> None:
    projection = EmbeddingMapResult(
        coordinates=np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        record_ids=["r1", "r2", "r3", "r4"],
        method="pca",
        config={"n_components": 2},
    )

    coverage = compute_design_space_coverage(projection)

    assert coverage["n_points"] == 4
    assert coverage["pairwise_spread"]["n_pairs"] == 6
    assert coverage["convex_hull"]["supported"] is True
    assert coverage["convex_hull"]["area"] == pytest.approx(1.0)
    assert coverage["config"]["input_source"] == "embedding_map_result"


def test_compute_design_space_coverage_handles_degenerate_cases() -> None:
    with pytest.raises(ValueError, match="2D matrix"):
        compute_design_space_coverage(np.asarray([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="at least one row"):
        compute_design_space_coverage(np.empty((0, 2)))

    three_dimensional = compute_design_space_coverage(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )
    )
    assert three_dimensional["convex_hull"]["area"] is None
    assert "only available for 2D inputs" in three_dimensional["warnings"][0]

    collinear = compute_design_space_coverage(
        np.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )
    )
    assert collinear["convex_hull"]["supported"] is False
    assert collinear["convex_hull"]["area"] is None


def test_compute_design_space_coverage_handles_embedding_results_and_single_points() -> None:
    embedding = EmbeddingResult(
        embeddings=np.asarray([[0.0, 1.0]]),
        record_ids=["r1"],
        texts=["alpha"],
        config={"provider": "callable"},
    )

    coverage = compute_design_space_coverage(embedding)

    assert coverage["config"]["input_source"] == "embedding_result"
    assert coverage["pairwise_spread"]["n_pairs"] == 0
    assert "fewer than two points" in coverage["warnings"][0]
    assert coverage["convex_hull"]["supported"] is False


def test_build_embedding_map_exposes_projection_compatibility_property() -> None:
    result = build_embedding_map(np.asarray([[0.0, 1.0], [1.0, 0.0]]), method="pca", n_components=2)

    assert np.array_equal(result.projection, result.coordinates)


def test_compute_design_space_coverage_rejects_nonfinite_values_and_unknown_methods() -> None:
    with pytest.raises(ValueError, match="finite numeric values"):
        compute_design_space_coverage(np.asarray([[0.0, 1.0], [np.inf, 2.0]]))
    with pytest.raises(ValueError, match="Unsupported method"):
        compute_design_space_coverage(np.asarray([[0.0, 1.0], [1.0, 2.0]]), method="alpha")


def test_embed_records_defaults_blank_record_ids_to_sorted_row_index() -> None:
    result = embed_records(
        [
            {"timestamp": "2026-01-01T10:00:01Z", "text": "later", "record_id": None},
            {"timestamp": "2026-01-01T10:00:00Z", "text": "first", "record_id": ""},
        ],
        embedder=lambda texts: np.asarray([[float(index)] for index, _ in enumerate(texts)]),
    )

    assert result.record_ids == ["0", "1"]


def test_compute_idea_space_trajectory_sorts_within_groups_by_timestamp() -> None:
    points = np.asarray(
        [
            [10.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
        ]
    )

    trajectory = compute_idea_space_trajectory(
        points,
        timestamps=[
            "2026-01-01T00:00:10Z",
            "2026-01-01T00:00:01Z",
            "2026-01-01T00:00:02Z",
            "2026-01-01T00:00:03Z",
        ],
        groups=["b", "a", "a", "a"],
    )

    assert trajectory["n_groups"] == 2
    assert trajectory["groups"]["a"]["ordered_indices"] == [1, 2, 3]
    assert trajectory["groups"]["a"]["step_sizes"] == [2.0, 1.0]
    assert trajectory["groups"]["a"]["path_length"] == pytest.approx(3.0)
    assert trajectory["groups"]["b"]["ordered_indices"] == [0]


def test_compute_idea_space_trajectory_defaults_to_global_group() -> None:
    trajectory = compute_idea_space_trajectory(np.asarray([[0.0, 0.0], [1.0, 0.0]]))

    assert trajectory["n_groups"] == 1
    assert list(trajectory["groups"]) == ["__all__"]
    assert trajectory["groups"]["__all__"]["path_length"] == pytest.approx(1.0)


def test_compute_idea_space_trajectory_handles_mixed_timestamp_types() -> None:
    trajectory = compute_idea_space_trajectory(
        np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ]
        ),
        timestamps=[
            datetime(2026, 1, 1, 10, 0, 0),
            np.int64(1),
            "3.5",
            "later",
            object(),
        ],
        groups=["session-a"] * 5,
    )

    timestamps = trajectory["groups"]["session-a"]["ordered_timestamps"]
    assert timestamps[0] == 1
    assert timestamps[1] == "3.5"
    assert str(timestamps[2]).startswith("2026-01-01T10:00:00")
    assert "later" in timestamps


def test_compute_idea_space_trajectory_validates_sequence_lengths() -> None:
    with pytest.raises(ValueError, match="timestamps must have the same length"):
        compute_idea_space_trajectory(
            np.asarray([[0.0, 0.0], [1.0, 1.0]]),
            timestamps=[0],
        )


def test_timestamp_helpers_cover_json_and_sorting_branches() -> None:
    assert _json_timestamp(datetime(2026, 1, 1, 10, 0, 0)).startswith("2026-01-01T10:00:00")
    assert _json_timestamp(np.int64(4)) == 4

    assert _timestamp_sort_key(None, index=0) == (2, "", 0)
    assert _timestamp_sort_key(datetime(2026, 1, 1, 10, 0, 0), index=1)[0] == 0
    assert _timestamp_sort_key(np.int64(2), index=2)[1] == 2.0
    assert _timestamp_sort_key("3.5", index=3)[1] == 3.5
    assert _timestamp_sort_key("later", index=4) == (1, "later", 4)
    assert _timestamp_sort_key(object(), index=5)[0] == 1


def test_compute_divergence_convergence_labels_rolling_phases() -> None:
    trajectory = {
        "groups": {
            "session-a": {
                "centroid_distances": [0.1, 0.4, 0.7, 0.5, 0.2],
            }
        }
    }

    summary = compute_divergence_convergence(trajectory, window=2)

    phases = [marker["phase"] for marker in summary["groups"]["session-a"]["phase_markers"]]
    assert phases == ["stable", "diverging", "diverging", "converging"]
    assert summary["groups"]["session-a"]["dominant_direction"] == "diverging"
    assert summary["groups"]["session-a"]["convergence_rate"] == pytest.approx(1 / 3)


def test_compute_divergence_convergence_covers_empty_short_and_stable_cases() -> None:
    summary = compute_divergence_convergence(
        {
            "groups": {
                "empty": {"centroid_distances": []},
                "short": {"centroid_distances": [0.2, 0.2, 0.2]},
            }
        },
        window=5,
    )

    assert "has no centroid distances" in summary["warnings"][0]
    assert summary["groups"]["empty"]["effective_window"] == 0
    assert summary["groups"]["short"]["effective_window"] == 3

    stable = compute_divergence_convergence(
        {"groups": {"flat": {"centroid_distances": [1.0, 1.0, 1.0, 1.0]}}},
        window=2,
    )
    assert stable["groups"]["flat"]["dominant_direction"] == "stable"


def test_compute_divergence_convergence_validates_inputs() -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        compute_divergence_convergence({"groups": {}}, window=0)
    with pytest.raises(ValueError, match="must include a 'groups' mapping"):
        compute_divergence_convergence({})
    with pytest.raises(ValueError, match="must be mappings"):
        compute_divergence_convergence({"groups": {"bad": []}})
