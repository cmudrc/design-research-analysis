"""Embedding and dimensionality-reduction utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from ._comparison import ComparableResultMixin
from .sequence.embeddings import embed_text
from .table import coerce_unified_table, derive_columns

_DIMRED_IMPORT_ERROR = (
    "This dimensionality-reduction method requires optional dependencies. "
    "Install with `pip install design-research-analysis[dimred]`."
)


@dataclass(slots=True)
class EmbeddingResult(ComparableResultMixin):
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

    def _comparison_metric(self) -> str:
        return "embedding_profile"

    def _comparison_vectors(
        self,
        other: EmbeddingResult,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return (
            self.embeddings.reshape(-1),
            other.embeddings.reshape(-1),
            {
                "left_shape": [int(dim) for dim in self.embeddings.shape],
                "right_shape": [int(dim) for dim in other.embeddings.shape],
                "left_record_ids": list(self.record_ids),
                "right_record_ids": list(other.record_ids),
            },
        )


@dataclass(slots=True)
class ProjectionResult(ComparableResultMixin):
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

    def _comparison_metric(self) -> str:
        return "projection_profile"

    def _comparison_vectors(
        self,
        other: ProjectionResult,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return (
            self.projection.reshape(-1),
            other.projection.reshape(-1),
            {
                "left_shape": [int(dim) for dim in self.projection.shape],
                "right_shape": [int(dim) for dim in other.projection.shape],
                "methods": [self.method, other.method],
            },
        )


def _coerce_feature_matrix(
    data: Sequence[Sequence[float]] | np.ndarray | EmbeddingResult | ProjectionResult,
    *,
    name: str,
) -> tuple[np.ndarray, str]:
    if isinstance(data, EmbeddingResult):
        matrix = np.asarray(data.embeddings, dtype=float)
        source = "embedding_result"
    elif isinstance(data, ProjectionResult):
        matrix = np.asarray(data.projection, dtype=float)
        source = "projection_result"
    else:
        matrix = np.asarray(data, dtype=float)
        source = "matrix"

    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite numeric values.")

    return matrix, source


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


def _pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.asarray([], dtype=float)
    row_ids, col_ids = np.triu_indices(matrix.shape[0], k=1)
    deltas = matrix[row_ids] - matrix[col_ids]
    return np.asarray(np.linalg.norm(deltas, axis=1), dtype=float)


def _summarize_distances(distances: np.ndarray) -> dict[str, float]:
    if distances.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
        }
    return {
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "std": float(np.std(distances)),
    }


def _centroid_radius_summary(matrix: np.ndarray) -> dict[str, Any]:
    centroid = np.mean(matrix, axis=0)
    distances = np.linalg.norm(matrix - centroid, axis=1)
    return {
        "centroid": centroid.astype(float).tolist(),
        "min_radius": float(np.min(distances)),
        "max_radius": float(np.max(distances)),
        "mean_radius": float(np.mean(distances)),
        "median_radius": float(np.median(distances)),
        "std_radius": float(np.std(distances)),
    }


def _cross_2d(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - origin[0]) * (b[1] - origin[1])) - (
        (a[1] - origin[1]) * (b[0] - origin[0])
    )


def _convex_hull_vertices(points: np.ndarray) -> list[tuple[float, float]]:
    unique_points = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique_points) <= 1:
        return unique_points

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _polygon_area(vertices: list[tuple[float, float]]) -> float:
    polygon = np.asarray(vertices, dtype=float)
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    return 0.5 * abs(
        float(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1)))
    )


def _polygon_perimeter(vertices: list[tuple[float, float]]) -> float:
    polygon = np.asarray(vertices, dtype=float)
    wrapped = np.vstack([polygon, polygon[0]])
    return float(np.sum(np.linalg.norm(np.diff(wrapped, axis=0), axis=1)))


def _convex_hull_summary(matrix: np.ndarray, warnings: list[str]) -> dict[str, Any]:
    if matrix.shape[1] != 2:
        warnings.append("Convex hull coverage is only available for 2D inputs.")
        return {
            "method": "convex_hull",
            "supported": False,
            "area": None,
            "volume": None,
            "perimeter": None,
            "n_vertices": 0,
        }

    unique_points = np.unique(matrix, axis=0)
    if unique_points.shape[0] < 3:
        warnings.append("Convex hull coverage requires at least three unique 2D points.")
        return {
            "method": "convex_hull",
            "supported": False,
            "area": None,
            "volume": None,
            "perimeter": None,
            "n_vertices": int(unique_points.shape[0]),
        }

    vertices = _convex_hull_vertices(unique_points)
    if len(vertices) < 3:
        warnings.append("Convex hull coverage is degenerate for collinear 2D points.")
        return {
            "method": "convex_hull",
            "supported": False,
            "area": None,
            "volume": None,
            "perimeter": None,
            "n_vertices": len(vertices),
        }

    area = _polygon_area(vertices)
    if area <= 0.0:
        warnings.append("Convex hull coverage is degenerate for collinear 2D points.")
        return {
            "method": "convex_hull",
            "supported": False,
            "area": None,
            "volume": None,
            "perimeter": None,
            "n_vertices": len(vertices),
        }

    return {
        "method": "convex_hull",
        "supported": True,
        "area": float(area),
        "volume": None,
        "perimeter": _polygon_perimeter(vertices),
        "n_vertices": len(vertices),
    }


def compute_design_space_coverage(
    embeddings: Sequence[Sequence[float]] | np.ndarray | EmbeddingResult | ProjectionResult,
    *,
    method: str = "convex_hull",
) -> dict[str, Any]:
    """Compute geometry-aware coverage summaries for embedding or projection spaces.

    Args:
        embeddings: Raw numeric matrix or an existing dimred result object.
        method: Hull coverage method. Currently supports ``"convex_hull"``.

    Returns:
        JSON-serializable coverage metrics.
    """
    if method.lower() != "convex_hull":
        raise ValueError("Unsupported method. Valid options: convex_hull.")

    matrix, source = _coerce_feature_matrix(embeddings, name="embeddings")
    warnings: list[str] = []
    if matrix.shape[0] < 2:
        warnings.append("Pairwise spread metrics are degenerate for fewer than two points.")

    pairwise = _pairwise_distances(matrix)
    return {
        "n_points": int(matrix.shape[0]),
        "n_dimensions": int(matrix.shape[1]),
        "pairwise_spread": {
            "n_pairs": int(pairwise.size),
            **_summarize_distances(pairwise),
        },
        "centroid_radius": _centroid_radius_summary(matrix),
        "convex_hull": _convex_hull_summary(matrix, warnings),
        "warnings": warnings,
        "config": {
            "input_source": source,
            "method": "convex_hull",
        },
    }


def _normalize_sequence(
    values: Sequence[Any] | None,
    *,
    length: int,
    label: str,
) -> list[Any] | None:
    if values is None:
        return None
    normalized = list(values)
    if len(normalized) != length:
        raise ValueError(f"{label} must have the same length as the input matrix.")
    return normalized


def _json_timestamp(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _timestamp_sort_key(value: Any, *, index: int) -> tuple[int, float | str, int]:
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
            try:
                return (0, float(stripped), index)
            except ValueError:
                return (1, stripped, index)
        return (0, float(parsed.timestamp()), index)
    return (1, str(value), index)


def compute_idea_space_trajectory(
    embeddings: Sequence[Sequence[float]] | np.ndarray | EmbeddingResult | ProjectionResult,
    *,
    timestamps: Sequence[Any] | None = None,
    groups: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compute grouped trajectories through an embedding or projection space.

    Args:
        embeddings: Raw numeric matrix or an existing dimred result object.
        timestamps: Optional sortable timestamps used within-group ordering.
        groups: Optional group labels such as session or condition.

    Returns:
        JSON-serializable per-group trajectory summaries.
    """
    matrix, source = _coerce_feature_matrix(embeddings, name="embeddings")
    timestamp_values = _normalize_sequence(timestamps, length=matrix.shape[0], label="timestamps")
    group_values = _normalize_sequence(groups, length=matrix.shape[0], label="groups")

    if group_values is None:
        normalized_groups = ["__all__"] * matrix.shape[0]
    else:
        normalized_groups = [
            "__all__" if _is_blank(group) else str(group) for group in group_values
        ]

    grouped_entries: dict[str, list[dict[str, Any]]] = {}
    for index, point in enumerate(matrix):
        timestamp = None if timestamp_values is None else timestamp_values[index]
        group = normalized_groups[index]
        grouped_entries.setdefault(group, []).append(
            {
                "index": index,
                "point": np.asarray(point, dtype=float),
                "timestamp": _json_timestamp(timestamp),
                "sort_key": _timestamp_sort_key(timestamp, index=index),
            }
        )

    groups_payload: dict[str, Any] = {}
    warnings: list[str] = []
    for group, entries in grouped_entries.items():
        ordered = sorted(entries, key=lambda item: item["sort_key"])
        ordered_points = np.vstack([entry["point"] for entry in ordered])
        centroid = np.mean(ordered_points, axis=0)
        centroid_distances = np.linalg.norm(ordered_points - centroid, axis=1)
        if ordered_points.shape[0] > 1:
            step_sizes = np.linalg.norm(np.diff(ordered_points, axis=0), axis=1)
            net_displacement = float(np.linalg.norm(ordered_points[-1] - ordered_points[0]))
        else:
            step_sizes = np.asarray([], dtype=float)
            net_displacement = 0.0
            warnings.append(f"Trajectory group '{group}' contains a single point.")

        groups_payload[group] = {
            "ordered_indices": [int(entry["index"]) for entry in ordered],
            "ordered_timestamps": [entry["timestamp"] for entry in ordered],
            "points": ordered_points.astype(float).tolist(),
            "centroid": centroid.astype(float).tolist(),
            "centroid_distances": centroid_distances.astype(float).tolist(),
            "step_sizes": step_sizes.astype(float).tolist(),
            "path_length": float(np.sum(step_sizes)),
            "net_displacement": net_displacement,
            "step_size_variance": float(np.var(step_sizes)) if step_sizes.size else 0.0,
            "n_points": int(ordered_points.shape[0]),
        }

    return {
        "n_points": int(matrix.shape[0]),
        "n_dimensions": int(matrix.shape[1]),
        "n_groups": len(groups_payload),
        "groups": groups_payload,
        "warnings": warnings,
        "config": {
            "input_source": source,
            "uses_provided_timestamps": timestamp_values is not None,
            "uses_provided_groups": group_values is not None,
        },
    }


def _rolling_means(values: list[float], *, window: int) -> list[float]:
    if not values:
        return []
    if window >= len(values):
        return [float(np.mean(values))]
    rolling: list[float] = []
    for start in range(0, len(values) - window + 1):
        rolling.append(float(np.mean(values[start : start + window])))
    return rolling


def compute_divergence_convergence(
    trajectory: Mapping[str, Any],
    *,
    window: int = 3,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Summarize divergence and convergence phases from trajectory output.

    Args:
        trajectory: Output from :func:`compute_idea_space_trajectory`.
        window: Rolling window size applied to centroid distances.
        tolerance: Threshold used to label stable changes.

    Returns:
        JSON-serializable divergence and convergence summaries by group.
    """
    if window <= 0:
        raise ValueError("window must be positive.")

    groups = trajectory.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError("trajectory must include a 'groups' mapping.")

    warnings: list[str] = []
    payload: dict[str, Any] = {}
    for group, raw_metrics in groups.items():
        if not isinstance(raw_metrics, Mapping):
            raise ValueError("trajectory group entries must be mappings.")
        centroid_distances_raw = raw_metrics.get("centroid_distances", [])
        centroid_distances = [float(value) for value in centroid_distances_raw]

        if not centroid_distances:
            warnings.append(f"Trajectory group '{group}' has no centroid distances.")
            payload[str(group)] = {
                "effective_window": 0,
                "rolling_mean_centroid_distance": [],
                "phase_markers": [],
                "divergence_score": 0.0,
                "convergence_rate": 0.0,
                "dominant_direction": "stable",
            }
            continue

        effective_window = min(window, len(centroid_distances))
        if effective_window < window:
            warnings.append(
                f"Trajectory group '{group}' is shorter than window {window}; "
                f"using window {effective_window}."
            )

        rolling = _rolling_means(centroid_distances, window=effective_window)
        labels = ["stable"]
        deltas: list[float] = []
        for index in range(1, len(rolling)):
            delta = rolling[index] - rolling[index - 1]
            deltas.append(delta)
            if delta > tolerance:
                labels.append("diverging")
            elif delta < -tolerance:
                labels.append("converging")
            else:
                labels.append("stable")

        counted_labels = labels[1:] if len(labels) > 1 else labels
        counts = {
            "diverging": counted_labels.count("diverging"),
            "converging": counted_labels.count("converging"),
            "stable": counted_labels.count("stable"),
        }
        max_count = max(counts.values()) if counts else 0
        dominant_candidates = [name for name, count in counts.items() if count == max_count]
        dominant_direction = "stable" if "stable" in dominant_candidates else dominant_candidates[0]

        positive_deltas = [delta for delta in deltas if delta > tolerance]
        convergence_count = sum(1 for delta in deltas if delta < -tolerance)
        phase_markers = [
            {
                "window_index": int(index),
                "start_step": int(index),
                "end_step": int(index + effective_window - 1),
                "phase": label,
                "rolling_mean_centroid_distance": float(rolling[index]),
            }
            for index, label in enumerate(labels)
        ]

        payload[str(group)] = {
            "effective_window": int(effective_window),
            "rolling_mean_centroid_distance": [float(value) for value in rolling],
            "phase_markers": phase_markers,
            "divergence_score": (
                float(np.mean(positive_deltas)) if positive_deltas else 0.0
            ),
            "convergence_rate": float(convergence_count / len(deltas)) if deltas else 0.0,
            "dominant_direction": dominant_direction,
        }

    return {
        "n_groups": len(payload),
        "groups": payload,
        "warnings": warnings,
        "config": {
            "window": int(window),
            "tolerance": float(tolerance),
        },
    }


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
    "compute_design_space_coverage",
    "compute_divergence_convergence",
    "compute_idea_space_trajectory",
    "embed_records",
    "reduce_dimensions",
]
