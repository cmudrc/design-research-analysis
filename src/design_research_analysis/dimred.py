"""Embedding and dimensionality-reduction utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .sequence.embeddings import embed_text
from .table import coerce_unified_table, derive_columns

_DIMRED_IMPORT_ERROR = (
    "This dimensionality-reduction method requires optional dependencies. "
    "Install with `pip install design-research-analysis[dimred]`."
)


@dataclass(slots=True)
class EmbeddingResult:
    """Embedding output container."""

    embeddings: np.ndarray
    record_ids: list[str]
    texts: list[str]
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result metadata to JSON-serializable format."""
        return {
            "shape": [int(dim) for dim in self.embeddings.shape],
            "record_ids": list(self.record_ids),
            "texts": list(self.texts),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class ProjectionResult:
    """Projection output container."""

    projection: np.ndarray
    method: str
    config: dict[str, Any] = field(default_factory=dict)
    explained_variance_ratio: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result metadata to JSON-serializable format."""
        return {
            "shape": [int(dim) for dim in self.projection.shape],
            "method": self.method,
            "explained_variance_ratio": (
                [float(value) for value in self.explained_variance_ratio]
                if self.explained_variance_ratio is not None
                else None
            ),
            "config": dict(self.config),
        }


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def embed_records(
    data: Sequence[Mapping[str, Any]],
    *,
    text_column: str = "text",
    record_id_column: str = "record_id",
    model_name: str = "all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 32,
    device: str = "auto",
    embedder: Callable[[Sequence[str]], np.ndarray] | None = None,
    text_mapper: Callable[[Mapping[str, Any]], Any] | None = None,
    record_id_mapper: Callable[[Mapping[str, Any]], Any] | None = None,
) -> EmbeddingResult:
    """Embed record text from a unified table.

    Args:
        data: Unified table rows.
        text_column: Text column name.
        record_id_column: Record identifier column name.
        model_name: SentenceTransformers model name for built-in embedding.
        normalize: Whether to normalize built-in embeddings.
        batch_size: Embedding batch size.
        device: Embedding device.
        embedder: Optional custom embedding function.
        text_mapper: Optional mapper to derive missing text.
        record_id_mapper: Optional mapper to derive missing record IDs.

    Returns:
        Embedding result containing vectors, record IDs, and source texts.
    """
    rows = coerce_unified_table(data)
    rows = derive_columns(
        rows,
        text_mapper=text_mapper,
        record_id_mapper=record_id_mapper,
    )

    texts: list[str] = []
    record_ids: list[str] = []
    for index, row in enumerate(rows):
        text = row.get(text_column)
        if _is_blank(text):
            raise ValueError(
                f"Row {index} is missing text in '{text_column}'. "
                "Provide text values or a text mapper."
            )
        record_id = row.get(record_id_column)
        if _is_blank(record_id):
            record_id = str(index)

        texts.append(str(text))
        record_ids.append(str(record_id))

    if not texts:
        raise ValueError("Embedding requires at least one text record.")

    if embedder is None:
        embedded = embed_text(
            texts,
            model_name=model_name,
            normalize=normalize,
            batch_size=batch_size,
            device=device,
        )
        provider = "sentence-transformers"
    else:
        embedded = np.asarray(embedder(texts), dtype=float)
        if embedded.ndim != 2:
            raise ValueError("embedder must return a 2D embedding matrix.")
        if embedded.shape[0] != len(texts):
            raise ValueError("embedder output rows must match number of text records.")
        provider = "callable"

    return EmbeddingResult(
        embeddings=np.asarray(embedded, dtype=float),
        record_ids=record_ids,
        texts=texts,
        config={
            "text_column": text_column,
            "record_id_column": record_id_column,
            "provider": provider,
            "model_name": model_name if embedder is None else "custom",
            "normalized_embeddings": bool(normalize),
            "batch_size": int(batch_size),
            "device": device,
        },
    )


def _pca_project(embeddings: np.ndarray, *, n_components: int) -> tuple[np.ndarray, list[float]]:
    centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    projected = u[:, :n_components] * s[:n_components]
    denom = np.sum(s**2)
    if denom == 0.0:
        explained = [0.0] * n_components
    else:
        explained = [(float(value**2) / float(denom)) for value in s[:n_components]]
    return projected, explained


def reduce_dimensions(
    embeddings: Sequence[Sequence[float]] | np.ndarray,
    *,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> ProjectionResult:
    """Reduce embedding dimensionality with deterministic controls.

    Supported methods: ``pca``, ``tsne``, ``umap``.
    """
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix.")
    if matrix.shape[0] == 0:
        raise ValueError("embeddings must contain at least one row.")
    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    method_key = method.lower()
    if method_key == "pca":
        projected, explained = _pca_project(matrix, n_components=n_components)
        return ProjectionResult(
            projection=np.asarray(projected, dtype=float),
            method="pca",
            explained_variance_ratio=explained,
            config={
                "n_components": int(n_components),
                "random_state": int(random_state),
            },
        )

    if method_key == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise ImportError(_DIMRED_IMPORT_ERROR) from exc

        tsne = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        )
        projected = tsne.fit_transform(matrix)
        return ProjectionResult(
            projection=np.asarray(projected, dtype=float),
            method="tsne",
            config={
                "n_components": int(n_components),
                "random_state": int(random_state),
                "perplexity": float(perplexity),
            },
        )

    if method_key == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(_DIMRED_IMPORT_ERROR) from exc

        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        projected = reducer.fit_transform(matrix)
        return ProjectionResult(
            projection=np.asarray(projected, dtype=float),
            method="umap",
            config={
                "n_components": int(n_components),
                "random_state": int(random_state),
                "n_neighbors": int(n_neighbors),
                "min_dist": float(min_dist),
            },
        )

    raise ValueError("Unsupported method. Valid options: pca, tsne, umap.")


def _kmeans(
    points: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    if n_clusters > points.shape[0]:
        raise ValueError("n_clusters cannot exceed number of points.")

    rng = np.random.default_rng(random_state)
    indices = rng.choice(points.shape[0], size=n_clusters, replace=False)
    centers = points[indices].copy()

    labels = np.zeros(points.shape[0], dtype=int)
    for _ in range(max_iter):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for cluster_id in range(n_clusters):
            members = points[labels == cluster_id]
            if len(members) == 0:
                centers[cluster_id] = points[rng.integers(0, points.shape[0])]
            else:
                centers[cluster_id] = np.mean(members, axis=0)

    return labels, centers


def cluster_projection(
    projection: Sequence[Sequence[float]] | np.ndarray,
    *,
    method: str = "kmeans",
    n_clusters: int = 3,
    random_state: int = 0,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Cluster projected points.

    Supported methods: ``kmeans`` (built-in), ``agglomerative`` (optional dependency).
    """
    points = np.asarray(projection, dtype=float)
    if points.ndim != 2:
        raise ValueError("projection must be a 2D matrix.")
    if points.shape[0] == 0:
        raise ValueError("projection must contain at least one row.")

    method_key = method.lower()
    if method_key == "kmeans":
        labels, centers = _kmeans(
            points,
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
        )
        return {
            "method": "kmeans",
            "labels": labels.astype(int).tolist(),
            "centers": centers.astype(float).tolist(),
            "config": {
                "n_clusters": int(n_clusters),
                "random_state": int(random_state),
                "max_iter": int(max_iter),
            },
        }

    if method_key == "agglomerative":
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError as exc:
            raise ImportError(_DIMRED_IMPORT_ERROR) from exc

        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(points)
        return {
            "method": "agglomerative",
            "labels": labels.astype(int).tolist(),
            "centers": None,
            "config": {"n_clusters": int(n_clusters)},
        }

    raise ValueError("Unsupported method. Valid options: kmeans, agglomerative.")


__all__ = [
    "EmbeddingResult",
    "ProjectionResult",
    "cluster_projection",
    "embed_records",
    "reduce_dimensions",
]
