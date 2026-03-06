from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from design_research_analysis.dimred import cluster_projection, reduce_dimensions


def test_reduce_dimensions_pca_is_deterministic() -> None:
    matrix = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=float,
    )

    first = reduce_dimensions(matrix, method="pca", n_components=2, random_state=2)
    second = reduce_dimensions(matrix, method="pca", n_components=2, random_state=2)

    assert np.allclose(first.projection, second.projection)
    assert first.explained_variance_ratio == second.explained_variance_ratio


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="sklearn unavailable")
def test_reduce_dimensions_tsne_deterministic_under_fixed_seed() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(30, 5))

    first = reduce_dimensions(
        matrix,
        method="tsne",
        n_components=2,
        random_state=7,
        perplexity=10.0,
    )
    second = reduce_dimensions(
        matrix,
        method="tsne",
        n_components=2,
        random_state=7,
        perplexity=10.0,
    )

    assert first.projection.shape == (30, 2)
    assert np.allclose(first.projection, second.projection)


@pytest.mark.skipif(importlib.util.find_spec("umap") is None, reason="umap unavailable")
def test_reduce_dimensions_umap_deterministic_under_fixed_seed() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(40, 6))

    first = reduce_dimensions(
        matrix,
        method="umap",
        n_components=2,
        random_state=11,
        n_neighbors=8,
    )
    second = reduce_dimensions(
        matrix,
        method="umap",
        n_components=2,
        random_state=11,
        n_neighbors=8,
    )

    assert np.allclose(first.projection, second.projection)


def test_cluster_projection_kmeans_is_deterministic() -> None:
    projection = np.asarray(
        [
            [0.0, 0.1],
            [0.1, 0.0],
            [5.0, 5.1],
            [5.2, 5.0],
            [10.0, 0.0],
            [10.1, 0.1],
        ]
    )

    first = cluster_projection(projection, method="kmeans", n_clusters=3, random_state=9)
    second = cluster_projection(projection, method="kmeans", n_clusters=3, random_state=9)

    assert first["labels"] == second["labels"]
    assert np.allclose(np.asarray(first["centers"]), np.asarray(second["centers"]))
