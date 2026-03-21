from __future__ import annotations

import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

import design_research_analysis.embedding_maps as maps_module
from design_research_analysis.embedding_maps import (
    EmbeddingMapResult,
    build_embedding_map,
    cluster_embedding_map,
    compare_embedding_maps,
    plot_embedding_map,
    plot_embedding_map_grid,
)


def test_build_embedding_map_pca_is_deterministic() -> None:
    matrix = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=float,
    )

    first = build_embedding_map(matrix, method="pca", n_components=2, random_state=2)
    second = build_embedding_map(matrix, method="pca", n_components=2, random_state=2)

    assert np.allclose(first.coordinates, second.coordinates)
    assert first.explained_variance_ratio == second.explained_variance_ratio


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="sklearn unavailable")
def test_build_embedding_map_tsne_deterministic_under_fixed_seed() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(30, 5))

    first = build_embedding_map(
        matrix,
        method="tsne",
        n_components=2,
        random_state=7,
        perplexity=10.0,
    )
    second = build_embedding_map(
        matrix,
        method="tsne",
        n_components=2,
        random_state=7,
        perplexity=10.0,
    )

    assert first.coordinates.shape == (30, 2)
    assert np.allclose(first.coordinates, second.coordinates)


@pytest.mark.skipif(importlib.util.find_spec("umap") is None, reason="umap unavailable")
def test_build_embedding_map_umap_deterministic_under_fixed_seed() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(40, 6))

    first = build_embedding_map(
        matrix,
        method="umap",
        n_components=2,
        random_state=11,
        n_neighbors=8,
    )
    second = build_embedding_map(
        matrix,
        method="umap",
        n_components=2,
        random_state=11,
        n_neighbors=8,
    )

    assert np.allclose(first.coordinates, second.coordinates)


@pytest.mark.skipif(importlib.util.find_spec("pacmap") is None, reason="pacmap unavailable")
def test_build_embedding_map_pacmap_deterministic_under_fixed_seed() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(24, 5))

    first = build_embedding_map(matrix, method="pacmap", n_components=2, random_state=5)
    second = build_embedding_map(matrix, method="pacmap", n_components=2, random_state=5)

    assert np.allclose(first.coordinates, second.coordinates)


@pytest.mark.skipif(importlib.util.find_spec("trimap") is None, reason="trimap unavailable")
def test_build_embedding_map_trimap_deterministic_under_fixed_seed() -> None:
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(24, 5))

    first = build_embedding_map(matrix, method="trimap", n_components=2, random_state=4)
    second = build_embedding_map(matrix, method="trimap", n_components=2, random_state=4)

    assert np.allclose(first.coordinates, second.coordinates)


def test_cluster_embedding_map_kmeans_is_deterministic() -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.1],
            [0.1, 0.0],
            [5.0, 5.1],
            [5.2, 5.0],
            [10.0, 0.0],
            [10.1, 0.1],
        ]
    )

    first = cluster_embedding_map(coordinates, method="kmeans", n_clusters=3, random_state=9)
    second = cluster_embedding_map(coordinates, method="kmeans", n_clusters=3, random_state=9)

    assert first["labels"] == second["labels"]
    assert np.allclose(np.asarray(first["centers"]), np.asarray(second["centers"]))


def test_compare_embedding_maps_preserves_method_order_and_record_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    def _fake_build(
        embeddings: np.ndarray,
        *,
        method: str,
        record_ids: list[str],
        **_: object,
    ) -> EmbeddingMapResult:
        assert embeddings.shape == (2, 2)
        return EmbeddingMapResult(
            coordinates=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float),
            record_ids=list(record_ids),
            method=method,
            config={"method": method},
        )

    monkeypatch.setattr(maps_module, "build_embedding_map", _fake_build)

    results = compare_embedding_maps(
        matrix,
        methods=["trimap", "pca"],
        record_ids=["r1", "r2"],
    )

    assert list(results) == ["trimap", "pca"]
    assert results["trimap"].record_ids == ["r1", "r2"]
    assert results["pca"].record_ids == ["r1", "r2"]


def test_plot_embedding_map_orders_traces_by_order_column() -> None:
    embedding_map = EmbeddingMapResult(
        coordinates=np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]], dtype=float),
        record_ids=["r1", "r2", "r3", "r4"],
        method="pca",
    )
    rows = [
        {"record_id": "r2", "trace_id": "t1", "step": "2"},
        {"record_id": "r1", "trace_id": "t1", "step": "1"},
        {"record_id": "r4", "trace_id": "t2", "step": "2"},
        {"record_id": "r3", "trace_id": "t2", "step": "1"},
    ]

    fig, ax = plot_embedding_map(
        embedding_map,
        rows,
        trace_column="trace_id",
        order_column="step",
    )

    assert len(fig.axes) == 1
    assert len(ax.lines) == 2
    assert list(ax.lines[0].get_xdata()) == [0.0, 1.0]
    assert list(ax.lines[1].get_xdata()) == [2.0, 3.0]
    plt.close(fig)


def test_plot_embedding_map_adds_value_colored_segments_and_colorbar() -> None:
    embedding_map = EmbeddingMapResult(
        coordinates=np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]], dtype=float),
        record_ids=["r1", "r2", "r3", "r4"],
        method="pca",
    )
    rows = [
        {"record_id": "r1", "trace_id": "t1", "step": "1", "value": "0.1"},
        {"record_id": "r2", "trace_id": "t1", "step": "2", "value": "0.5"},
        {"record_id": "r3", "trace_id": "t2", "step": "1", "value": "0.2"},
        {"record_id": "r4", "trace_id": "t2", "step": "2", "value": "0.8"},
    ]

    fig, ax = plot_embedding_map(
        embedding_map,
        rows,
        trace_column="trace_id",
        order_column="step",
        value_column="value",
    )

    line_collections = [item for item in ax.collections if isinstance(item, LineCollection)]
    assert len(fig.axes) == 2
    assert len(line_collections) == 2
    assert all(collection.get_array() is not None for collection in line_collections)
    plt.close(fig)


def test_plot_embedding_map_grid_shares_value_norm_across_maps() -> None:
    rows = [
        {"record_id": "r1", "trace_id": "t1", "step": "1", "value": "0.1"},
        {"record_id": "r2", "trace_id": "t1", "step": "2", "value": "0.5"},
        {"record_id": "r3", "trace_id": "t2", "step": "1", "value": "0.2"},
        {"record_id": "r4", "trace_id": "t2", "step": "2", "value": "0.8"},
    ]
    first = EmbeddingMapResult(
        coordinates=np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]], dtype=float),
        record_ids=["r1", "r2", "r3", "r4"],
        method="pca",
    )
    second = EmbeddingMapResult(
        coordinates=np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 1.0], [3.0, 2.0]], dtype=float),
        record_ids=["r1", "r2", "r3", "r4"],
        method="umap",
    )

    fig, axes = plot_embedding_map_grid(
        {"pca": first, "umap": second},
        rows,
        trace_column="trace_id",
        order_column="step",
        value_column="value",
    )

    first_lines = [item for item in axes[0].collections if isinstance(item, LineCollection)]
    second_lines = [item for item in axes[1].collections if isinstance(item, LineCollection)]

    assert len(axes) == 2
    assert len(fig.axes) == 3
    assert first_lines[0].norm is second_lines[0].norm
    plt.close(fig)
