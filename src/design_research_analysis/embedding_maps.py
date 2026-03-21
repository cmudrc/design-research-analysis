"""Embedding-map and idea-space analysis utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from ._comparison import ComparableResultMixin
from .sequence.embeddings import embed_text
from .table import coerce_unified_table, derive_columns

_MAPS_IMPORT_ERROR = (
    "This embedding-map method requires optional dependencies. "
    "Install with `pip install design-research-analysis[maps]`."
)

_METHOD_DISPLAY_NAMES = {
    "pca": "PCA",
    "tsne": "t-SNE",
    "umap": "UMAP",
    "pacmap": "PaCMAP",
    "trimap": "TriMap",
}


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
class EmbeddingMapResult(ComparableResultMixin):
    """Lower-dimensional coordinates plus method metadata for one embedding map."""

    coordinates: np.ndarray
    record_ids: list[str]
    method: str
    config: dict[str, Any] = field(default_factory=dict)
    explained_variance_ratio: list[float] | None = None

    @property
    def projection(self) -> np.ndarray:
        """Compatibility alias for the legacy projection attribute name."""
        return self.coordinates

    def to_dict(self) -> dict[str, Any]:
        """Convert result metadata to JSON-serializable format."""
        return {
            "shape": [int(dim) for dim in self.coordinates.shape],
            "record_ids": list(self.record_ids),
            "method": self.method,
            "explained_variance_ratio": (
                [float(value) for value in self.explained_variance_ratio]
                if self.explained_variance_ratio is not None
                else None
            ),
            "config": dict(self.config),
        }

    def _comparison_metric(self) -> str:
        return "embedding_map_profile"

    def _comparison_vectors(
        self,
        other: EmbeddingMapResult,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return (
            self.coordinates.reshape(-1),
            other.coordinates.reshape(-1),
            {
                "left_shape": [int(dim) for dim in self.coordinates.shape],
                "right_shape": [int(dim) for dim in other.coordinates.shape],
                "methods": [self.method, other.method],
                "left_record_ids": list(self.record_ids),
                "right_record_ids": list(other.record_ids),
            },
        )


def _coerce_feature_matrix(
    data: Sequence[Sequence[float]] | np.ndarray | EmbeddingResult | EmbeddingMapResult,
    *,
    name: str,
) -> tuple[np.ndarray, str]:
    if isinstance(data, EmbeddingResult):
        matrix = np.asarray(data.embeddings, dtype=float)
        source = "embedding_result"
    elif isinstance(data, EmbeddingMapResult):
        matrix = np.asarray(data.coordinates, dtype=float)
        source = "embedding_map_result"
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


def _method_display_name(method: str) -> str:
    return _METHOD_DISPLAY_NAMES.get(method.lower(), method)


def _resolve_record_ids(record_ids: Sequence[Any] | None, n_rows: int) -> list[str]:
    if record_ids is None:
        resolved = [str(index) for index in range(n_rows)]
    else:
        if len(record_ids) != n_rows:
            raise ValueError("record_ids length must match the number of rows.")
        resolved = [str(record_id) for record_id in record_ids]

    if len(set(resolved)) != len(resolved):
        raise ValueError("record_ids must be unique.")
    return resolved


def _pca_project(matrix: np.ndarray, *, n_components: int) -> tuple[np.ndarray, list[float]]:
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
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
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _convex_hull_vertices(points: np.ndarray) -> list[tuple[float, float]]:
    unique_points = sorted({(float(x), float(y)) for x, y in points.tolist()})
    if len(unique_points) <= 1:
        return unique_points

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    if hull and hull[0] == hull[-1]:
        hull.pop()
    return hull


def _polygon_area(vertices: list[tuple[float, float]]) -> float:
    if len(vertices) < 3:
        return 0.0
    area = 0.0
    for index, (x_coord, y_coord) in enumerate(vertices):
        next_x, next_y = vertices[(index + 1) % len(vertices)]
        area += x_coord * next_y - next_x * y_coord
    return abs(area) * 0.5


def _polygon_perimeter(vertices: list[tuple[float, float]]) -> float:
    if len(vertices) < 2:
        return 0.0
    distances = []
    for index, point in enumerate(vertices):
        next_point = vertices[(index + 1) % len(vertices)]
        distances.append(np.linalg.norm(np.asarray(point) - np.asarray(next_point)))
    return float(np.sum(distances))


def _convex_hull_summary(matrix: np.ndarray, warnings: list[str]) -> dict[str, Any]:
    if matrix.shape[1] != 2:
        warnings.append("Convex-hull area is only available for 2D inputs.")
        return {
            "supported": False,
            "area": None,
            "perimeter": None,
            "n_vertices": 0,
        }

    vertices = _convex_hull_vertices(matrix)
    if len(vertices) < 3:
        warnings.append("Convex hull is degenerate for fewer than three non-collinear points.")
        return {
            "supported": False,
            "area": None,
            "perimeter": None,
            "n_vertices": len(vertices),
        }

    area = _polygon_area(vertices)
    if area == 0.0:
        warnings.append("Convex hull area is zero because all points are collinear.")
        return {
            "supported": False,
            "area": None,
            "perimeter": _polygon_perimeter(vertices),
            "n_vertices": len(vertices),
        }

    return {
        "supported": True,
        "area": area,
        "perimeter": _polygon_perimeter(vertices),
        "n_vertices": len(vertices),
    }


def _run_with_numpy_seed(seed: int, func: Callable[[], np.ndarray]) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return np.asarray(func(), dtype=float)
    finally:
        np.random.set_state(state)


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
    """Embed record text from a unified table."""
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

    if len(set(record_ids)) != len(record_ids):
        raise ValueError("record_id values must be unique for embedding maps.")

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


def compute_design_space_coverage(
    embeddings: Sequence[Sequence[float]] | np.ndarray | EmbeddingResult | EmbeddingMapResult,
    *,
    method: str = "convex_hull",
) -> dict[str, Any]:
    """Compute geometry-aware coverage summaries for embedding or map spaces."""
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
    embeddings: Sequence[Sequence[float]] | np.ndarray | EmbeddingResult | EmbeddingMapResult,
    *,
    timestamps: Sequence[Any] | None = None,
    groups: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compute grouped trajectories through an embedding or map space."""
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
    """Summarize divergence and convergence phases from trajectory output."""
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
            "divergence_score": (float(np.mean(positive_deltas)) if positive_deltas else 0.0),
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


def build_embedding_map(
    embeddings: Sequence[Sequence[float]] | np.ndarray,
    *,
    method: str = "pca",
    n_components: int = 2,
    record_ids: Sequence[Any] | None = None,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    pacmap_mn_ratio: float = 0.5,
    pacmap_fp_ratio: float = 2.0,
    trimap_n_inliers: int = 12,
    trimap_n_outliers: int = 4,
    trimap_n_random: int = 3,
) -> EmbeddingMapResult:
    """Map higher-dimensional vectors into a lower-dimensional embedding space."""
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix.")
    if matrix.shape[0] == 0:
        raise ValueError("embeddings must contain at least one row.")
    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    resolved_record_ids = _resolve_record_ids(record_ids, matrix.shape[0])
    method_key = method.lower()

    if method_key == "pca":
        coordinates, explained = _pca_project(matrix, n_components=n_components)
        return EmbeddingMapResult(
            coordinates=np.asarray(coordinates, dtype=float),
            record_ids=resolved_record_ids,
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
            raise ImportError(_MAPS_IMPORT_ERROR) from exc

        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        )
        coordinates = reducer.fit_transform(matrix)
        return EmbeddingMapResult(
            coordinates=np.asarray(coordinates, dtype=float),
            record_ids=resolved_record_ids,
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
            raise ImportError(_MAPS_IMPORT_ERROR) from exc

        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        coordinates = reducer.fit_transform(matrix)
        return EmbeddingMapResult(
            coordinates=np.asarray(coordinates, dtype=float),
            record_ids=resolved_record_ids,
            method="umap",
            config={
                "n_components": int(n_components),
                "random_state": int(random_state),
                "n_neighbors": int(n_neighbors),
                "min_dist": float(min_dist),
            },
        )

    if method_key == "pacmap":
        try:
            import pacmap
        except ImportError as exc:
            raise ImportError(_MAPS_IMPORT_ERROR) from exc

        reducer = pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            MN_ratio=pacmap_mn_ratio,
            FP_ratio=pacmap_fp_ratio,
        )
        coordinates = _run_with_numpy_seed(
            random_state,
            lambda: reducer.fit_transform(matrix, init="pca"),
        )
        return EmbeddingMapResult(
            coordinates=np.asarray(coordinates, dtype=float),
            record_ids=resolved_record_ids,
            method="pacmap",
            config={
                "n_components": int(n_components),
                "random_state": int(random_state),
                "n_neighbors": int(n_neighbors),
                "mn_ratio": float(pacmap_mn_ratio),
                "fp_ratio": float(pacmap_fp_ratio),
            },
        )

    if method_key == "trimap":
        try:
            import trimap
        except ImportError as exc:
            raise ImportError(_MAPS_IMPORT_ERROR) from exc

        reducer = trimap.TRIMAP(
            n_dims=n_components,
            n_inliers=trimap_n_inliers,
            n_outliers=trimap_n_outliers,
            n_random=trimap_n_random,
        )
        coordinates = _run_with_numpy_seed(random_state, lambda: reducer.fit_transform(matrix))
        return EmbeddingMapResult(
            coordinates=np.asarray(coordinates, dtype=float),
            record_ids=resolved_record_ids,
            method="trimap",
            config={
                "n_components": int(n_components),
                "random_state": int(random_state),
                "n_inliers": int(trimap_n_inliers),
                "n_outliers": int(trimap_n_outliers),
                "n_random": int(trimap_n_random),
            },
        )

    raise ValueError("Unsupported method. Valid options: pca, tsne, umap, pacmap, trimap.")


def compare_embedding_maps(
    embeddings: Sequence[Sequence[float]] | np.ndarray,
    *,
    methods: Sequence[str],
    n_components: int = 2,
    record_ids: Sequence[Any] | None = None,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    pacmap_mn_ratio: float = 0.5,
    pacmap_fp_ratio: float = 2.0,
    trimap_n_inliers: int = 12,
    trimap_n_outliers: int = 4,
    trimap_n_random: int = 3,
) -> dict[str, EmbeddingMapResult]:
    """Build multiple embedding maps with aligned record IDs."""
    method_list = [str(method).lower() for method in methods]
    if not method_list:
        raise ValueError("methods must contain at least one mapping method.")
    if len(set(method_list)) != len(method_list):
        raise ValueError("methods must be unique.")

    compared: dict[str, EmbeddingMapResult] = {}
    for method_name in method_list:
        compared[method_name] = build_embedding_map(
            embeddings,
            method=method_name,
            n_components=n_components,
            record_ids=record_ids,
            random_state=random_state,
            perplexity=perplexity,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            pacmap_mn_ratio=pacmap_mn_ratio,
            pacmap_fp_ratio=pacmap_fp_ratio,
            trimap_n_inliers=trimap_n_inliers,
            trimap_n_outliers=trimap_n_outliers,
            trimap_n_random=trimap_n_random,
        )
    return compared


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


def cluster_embedding_map(
    embedding_map: EmbeddingMapResult | Sequence[Sequence[float]] | np.ndarray,
    *,
    method: str = "kmeans",
    n_clusters: int = 3,
    random_state: int = 0,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Cluster embedding-map coordinates."""
    points = (
        np.asarray(embedding_map.coordinates, dtype=float)
        if isinstance(embedding_map, EmbeddingMapResult)
        else np.asarray(embedding_map, dtype=float)
    )
    if points.ndim != 2:
        raise ValueError("embedding_map must be a 2D matrix.")
    if points.shape[0] == 0:
        raise ValueError("embedding_map must contain at least one row.")

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
            raise ImportError(_MAPS_IMPORT_ERROR) from exc

        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(points)
        return {
            "method": "agglomerative",
            "labels": labels.astype(int).tolist(),
            "centers": None,
            "config": {"n_clusters": int(n_clusters)},
        }

    raise ValueError("Unsupported method. Valid options: kmeans, agglomerative.")


def _sort_key(value: Any) -> tuple[int, float | str]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value))
    if isinstance(value, str):
        text = value.strip()
        try:
            return (0, float(text))
        except ValueError:
            return (1, text)
    return (1, str(value))


def _value_norm(values: Sequence[float]) -> Normalize:
    minimum = float(min(values))
    maximum = float(max(values))
    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5
    return Normalize(vmin=minimum, vmax=maximum)


def _indexed_rows(
    data: Sequence[Mapping[str, Any]],
    *,
    record_id_column: str,
) -> dict[str, Mapping[str, Any]]:
    rows = coerce_unified_table(data)
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        record_id = row.get(record_id_column)
        if _is_blank(record_id):
            raise ValueError(
                f"Row {index} is missing '{record_id_column}'. "
                "Embedding-map plotting requires record IDs."
            )
        record_key = str(record_id)
        if record_key in indexed:
            raise ValueError(f"Duplicate record_id '{record_key}' in plotting data.")
        indexed[record_key] = row
    return indexed


def _joined_rows(
    embedding_map: EmbeddingMapResult,
    data: Sequence[Mapping[str, Any]],
    *,
    record_id_column: str,
) -> list[Mapping[str, Any]]:
    indexed = _indexed_rows(data, record_id_column=record_id_column)
    joined: list[Mapping[str, Any]] = []
    missing: list[str] = []
    for record_id in embedding_map.record_ids:
        row = indexed.get(record_id)
        if row is None:
            missing.append(record_id)
        else:
            joined.append(row)
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(f"Plotting data is missing map record IDs: {sample}")
    return joined


def _matched_values(joined_rows: Sequence[Mapping[str, Any]], *, value_column: str) -> np.ndarray:
    values: list[float] = []
    for index, row in enumerate(joined_rows):
        raw_value = row.get(value_column)
        if _is_blank(raw_value):
            raise ValueError(f"Row {index} is missing '{value_column}'.")
        try:
            values.append(float(cast(float | int | str, raw_value)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Row {index} has non-numeric '{value_column}'.") from exc
    return np.asarray(values, dtype=float)


def _trace_groups(
    embedding_map: EmbeddingMapResult,
    joined_rows: Sequence[Mapping[str, Any]],
    *,
    trace_column: str,
    order_column: str,
    value_column: str | None,
) -> dict[str, list[tuple[tuple[int, float | str], np.ndarray, float | None]]]:
    groups: dict[str, list[tuple[tuple[int, float | str], np.ndarray, float | None]]] = {}
    for record_id, row, coordinate in zip(
        embedding_map.record_ids,
        joined_rows,
        embedding_map.coordinates,
        strict=True,
    ):
        trace_value = row.get(trace_column)
        order_value = row.get(order_column)
        if _is_blank(trace_value):
            raise ValueError(f"Row for record_id '{record_id}' is missing '{trace_column}'.")
        if _is_blank(order_value):
            raise ValueError(f"Row for record_id '{record_id}' is missing '{order_column}'.")
        scalar_value: float | None = None
        if value_column is not None:
            raw_value = row.get(value_column)
            if _is_blank(raw_value):
                raise ValueError(f"Row for record_id '{record_id}' is missing '{value_column}'.")
            try:
                scalar_value = float(cast(float | int | str, raw_value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Row for record_id '{record_id}' has non-numeric '{value_column}'."
                ) from exc

        groups.setdefault(str(trace_value), []).append(
            (_sort_key(order_value), np.asarray(coordinate, dtype=float), scalar_value)
        )

    for trace_id in groups:
        groups[trace_id].sort(key=lambda item: item[0])
    return groups


def _plot_trace_overlays(
    ax: Axes,
    embedding_map: EmbeddingMapResult,
    joined_rows: Sequence[Mapping[str, Any]],
    *,
    trace_column: str | None,
    order_column: str | None,
    value_column: str | None,
    cmap: str,
    norm: Normalize | None,
) -> None:
    if trace_column is None:
        return
    if order_column is None:
        raise ValueError("order_column is required when trace_column is provided.")

    groups = _trace_groups(
        embedding_map,
        joined_rows,
        trace_column=trace_column,
        order_column=order_column,
        value_column=value_column,
    )
    palette = plt.get_cmap("tab10", max(1, len(groups)))

    for trace_index, (trace_id, items) in enumerate(groups.items()):
        points = np.asarray([item[1] for item in items], dtype=float)
        if value_column is not None and norm is not None:
            trace_values = np.asarray(
                [0.0 if item[2] is None else float(item[2]) for item in items],
                dtype=float,
            )
            if len(points) > 1:
                segments = np.stack([points[:-1], points[1:]], axis=1)
                segment_values = (trace_values[:-1] + trace_values[1:]) / 2.0
                collection = LineCollection(
                    segments.tolist(),
                    cmap=plt.get_cmap(cmap),
                    norm=norm,
                    linewidths=2.0,
                    alpha=0.95,
                )
                collection.set_array(segment_values)
                ax.add_collection(collection)
            ax.scatter(
                [points[0, 0]],
                [points[0, 1]],
                c=[trace_values[0]],
                cmap=cmap,
                norm=norm,
                marker="o",
                s=60,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
            ax.scatter(
                [points[-1, 0]],
                [points[-1, 1]],
                c=[trace_values[-1]],
                cmap=cmap,
                norm=norm,
                marker="X",
                s=70,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
        else:
            color = palette(trace_index)
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                linewidth=2.0,
                alpha=0.9,
                label=trace_id,
                zorder=3,
            )
            ax.scatter(
                [points[0, 0]],
                [points[0, 1]],
                color=[color],
                marker="o",
                s=55,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
            ax.scatter(
                [points[-1, 0]],
                [points[-1, 1]],
                color=[color],
                marker="X",
                s=65,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )


def _plot_embedding_map_on_axis(
    embedding_map: EmbeddingMapResult,
    data: Sequence[Mapping[str, Any]],
    *,
    record_id_column: str,
    trace_column: str | None,
    order_column: str | None,
    value_column: str | None,
    ax: Axes,
    cmap: str,
    norm: Normalize | None,
    add_colorbar: bool,
    title: str | None,
) -> None:
    if embedding_map.coordinates.ndim != 2 or embedding_map.coordinates.shape[1] != 2:
        raise ValueError("plotting requires a 2D embedding map.")

    joined_rows = _joined_rows(embedding_map, data, record_id_column=record_id_column)
    values = (
        _matched_values(joined_rows, value_column=value_column)
        if value_column is not None
        else None
    )

    if values is not None and norm is None:
        norm = _value_norm(values.tolist())

    if values is not None and norm is not None:
        scatter = ax.scatter(
            embedding_map.coordinates[:, 0],
            embedding_map.coordinates[:, 1],
            c=values,
            cmap=cmap,
            norm=norm,
            s=26,
            alpha=0.85,
            linewidths=0.0,
            zorder=2,
        )
    else:
        scatter = ax.scatter(
            embedding_map.coordinates[:, 0],
            embedding_map.coordinates[:, 1],
            color="#c9d1d9",
            s=24,
            alpha=0.55,
            linewidths=0.0,
            zorder=1,
        )

    _plot_trace_overlays(
        ax,
        embedding_map,
        joined_rows,
        trace_column=trace_column,
        order_column=order_column,
        value_column=value_column,
        cmap=cmap,
        norm=norm,
    )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title or _method_display_name(embedding_map.method))
    ax.grid(alpha=0.18)

    if add_colorbar and values is not None and norm is not None:
        mappable = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
        mappable.set_array(values)
        cast(Figure, ax.figure).colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    _ = scatter


def plot_embedding_map(
    embedding_map: EmbeddingMapResult,
    data: Sequence[Mapping[str, Any]],
    *,
    record_id_column: str = "record_id",
    trace_column: str | None = None,
    order_column: str | None = None,
    value_column: str | None = None,
    ax: Axes | None = None,
    cmap: str = "viridis",
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot one embedding map with optional trace overlays and scalar coloring."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = cast(Figure, ax.figure)

    norm = None
    if value_column is not None:
        joined_rows = _joined_rows(embedding_map, data, record_id_column=record_id_column)
        norm = _value_norm(_matched_values(joined_rows, value_column=value_column).tolist())

    _plot_embedding_map_on_axis(
        embedding_map,
        data,
        record_id_column=record_id_column,
        trace_column=trace_column,
        order_column=order_column,
        value_column=value_column,
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=True,
        title=title,
    )
    return fig, ax


def plot_embedding_map_grid(
    embedding_maps: Mapping[str, EmbeddingMapResult],
    data: Sequence[Mapping[str, Any]],
    *,
    record_id_column: str = "record_id",
    trace_column: str | None = None,
    order_column: str | None = None,
    value_column: str | None = None,
    cmap: str = "viridis",
    title: str = "Embedding Map Comparison",
) -> tuple[Figure, list[Axes]]:
    """Plot multiple embedding maps with shared trace overlays and color scale."""
    if not embedding_maps:
        raise ValueError("embedding_maps must contain at least one map.")

    items = list(embedding_maps.items())
    fig, axes = plt.subplots(1, len(items), figsize=(6 * len(items), 5), squeeze=False)
    flat_axes = [cast(Axes, axis) for axis in axes.reshape(-1)]

    norm = None
    if value_column is not None:
        all_values: list[float] = []
        for _, embedding_map in items:
            joined_rows = _joined_rows(embedding_map, data, record_id_column=record_id_column)
            all_values.extend(_matched_values(joined_rows, value_column=value_column).tolist())
        norm = _value_norm(all_values)

    for axis, (method_name, embedding_map) in zip(flat_axes, items, strict=True):
        _plot_embedding_map_on_axis(
            embedding_map,
            data,
            record_id_column=record_id_column,
            trace_column=trace_column,
            order_column=order_column,
            value_column=value_column,
            ax=axis,
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
            title=_method_display_name(method_name),
        )

    fig.suptitle(title)
    fig.tight_layout()

    if value_column is not None and norm is not None:
        mappable = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
        fig.colorbar(mappable, ax=flat_axes, fraction=0.02, pad=0.04)

    return fig, flat_axes


__all__ = [
    "EmbeddingMapResult",
    "EmbeddingResult",
    "build_embedding_map",
    "cluster_embedding_map",
    "compare_embedding_maps",
    "compute_design_space_coverage",
    "compute_divergence_convergence",
    "compute_idea_space_trajectory",
    "embed_records",
    "plot_embedding_map",
    "plot_embedding_map_grid",
]
