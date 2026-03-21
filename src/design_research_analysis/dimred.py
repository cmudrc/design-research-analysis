"""Compatibility wrappers for legacy dimensionality-reduction imports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .embedding_maps import (
    EmbeddingMapResult,
    EmbeddingResult,
    _coerce_feature_matrix,
    _json_timestamp,
    _kmeans,
    _timestamp_sort_key,
    build_embedding_map,
    cluster_embedding_map,
    compute_design_space_coverage,
    compute_divergence_convergence,
    compute_idea_space_trajectory,
    embed_records,
)

ProjectionResult = EmbeddingMapResult


def reduce_dimensions(
    embeddings: Sequence[Sequence[float]] | np.ndarray,
    *,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> EmbeddingMapResult:
    """Legacy wrapper for :func:`build_embedding_map`."""
    return build_embedding_map(
        embeddings,
        method=method,
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )


def cluster_projection(
    projection: Sequence[Sequence[float]] | np.ndarray | EmbeddingMapResult,
    *,
    method: str = "kmeans",
    n_clusters: int = 3,
    random_state: int = 0,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Legacy wrapper for :func:`cluster_embedding_map`."""
    return cluster_embedding_map(
        projection,
        method=method,
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=max_iter,
    )


__all__ = [
    "EmbeddingResult",
    "ProjectionResult",
    "_coerce_feature_matrix",
    "_json_timestamp",
    "_kmeans",
    "_timestamp_sort_key",
    "cluster_projection",
    "compute_design_space_coverage",
    "compute_divergence_convergence",
    "compute_idea_space_trajectory",
    "embed_records",
    "reduce_dimensions",
]
