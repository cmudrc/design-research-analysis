"""Language analysis utilities for convergence, topics, and sentiment."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeGuard

import numpy as np

from .sequence.embeddings import embed_text
from .table import coerce_unified_table, derive_columns

_LANG_IMPORT_ERROR = (
    "Topic modeling requires optional language dependencies. "
    "Install with `pip install design-research-analysis[lang]`."
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z']+")

_POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "better",
    "best",
    "clear",
    "helpful",
    "improve",
    "improved",
    "improvement",
    "success",
    "successful",
    "positive",
    "creative",
    "efficient",
    "effective",
    "confident",
    "collaborative",
    "useful",
    "strong",
}

_NEGATIVE_WORDS = {
    "bad",
    "worse",
    "worst",
    "unclear",
    "confusing",
    "difficult",
    "hard",
    "frustrated",
    "frustrating",
    "error",
    "errors",
    "problem",
    "problems",
    "negative",
    "stress",
    "stressed",
    "risky",
    "risk",
    "slow",
    "weak",
}


@dataclass(slots=True)
class LanguageConvergenceResult:
    """Result container for language convergence/divergence analysis."""

    groups: list[str]
    distance_trajectories: dict[str, list[float]]
    slope_by_group: dict[str, float]
    direction_by_group: dict[str, str]
    window_size: int
    n_observations: int
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "groups": list(self.groups),
            "distance_trajectories": {k: list(v) for k, v in self.distance_trajectories.items()},
            "slope_by_group": {k: float(v) for k, v in self.slope_by_group.items()},
            "direction_by_group": dict(self.direction_by_group),
            "window_size": int(self.window_size),
            "n_observations": int(self.n_observations),
            "config": dict(self.config),
        }


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _is_text_sequence(
    data: Sequence[Mapping[str, Any]] | Sequence[str],
) -> TypeGuard[Sequence[str]]:
    return len(data) > 0 and isinstance(data[0], str)


def _extract_text_rows(
    data: Sequence[Mapping[str, Any]] | Sequence[str],
    *,
    text_column: str,
    group_column: str,
    text_mapper: Callable[[Mapping[str, Any]], Any] | None,
) -> tuple[list[str], list[str], int]:
    if _is_text_sequence(data):
        text_items = [str(item) for item in data]
        return text_items, ["__all__"] * len(text_items), len(text_items)

    rows = coerce_unified_table(data)
    rows = derive_columns(rows, text_mapper=text_mapper)

    texts: list[str] = []
    groups: list[str] = []
    for index, row in enumerate(rows):
        text = row.get(text_column)
        if _is_blank(text):
            raise ValueError(
                f"Row {index} is missing text in '{text_column}'. "
                "Provide text values or a text mapper."
            )
        group = row.get(group_column)
        texts.append(str(text))
        groups.append("__all__" if _is_blank(group) else str(group))
    return texts, groups, len(rows)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    similarity = float(np.dot(a, b) / (a_norm * b_norm))
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def compute_semantic_distance_trajectory(
    data: Sequence[Mapping[str, Any]] | Sequence[str],
    *,
    text_column: str = "text",
    group_column: str = "session_id",
    window_size: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 32,
    device: str = "auto",
    embedder: Callable[[Sequence[str]], np.ndarray] | None = None,
    text_mapper: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, list[float]]:
    """Compute semantic distance trajectories to a group's final language state.

    Args:
        data: Unified table rows or a simple text list.
        text_column: Text column for table input.
        group_column: Grouping column for trajectory computation.
        window_size: Sliding window size for centroid estimation.
        model_name: Sentence transformer model name when ``embedder`` is omitted.
        normalize: Whether to normalize embeddings when using built-in embedding.
        batch_size: Embedding batch size.
        device: Embedding device.
        embedder: Optional custom embedding function.
        text_mapper: Optional mapper used to derive missing text values.

    Returns:
        Mapping of ``group -> [distance_t0, distance_t1, ...]``.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    texts, groups, _ = _extract_text_rows(
        data,
        text_column=text_column,
        group_column=group_column,
        text_mapper=text_mapper,
    )

    if embedder is None:
        embedded = embed_text(
            texts,
            model_name=model_name,
            normalize=normalize,
            batch_size=batch_size,
            device=device,
        )
    else:
        embedded = np.asarray(embedder(texts), dtype=float)
        if embedded.ndim != 2:
            raise ValueError("embedder must return a 2D embedding matrix.")
        if embedded.shape[0] != len(texts):
            raise ValueError("embedder output rows must match number of texts.")

    by_group: dict[str, list[np.ndarray]] = {}
    for group, vector in zip(groups, embedded, strict=True):
        by_group.setdefault(group, []).append(np.asarray(vector, dtype=float))

    trajectories: dict[str, list[float]] = {}
    for group, vectors in by_group.items():
        if len(vectors) < window_size:
            trajectories[group] = []
            continue

        windows: list[np.ndarray] = []
        for start in range(0, len(vectors) - window_size + 1):
            window_matrix = np.vstack(vectors[start : start + window_size])
            windows.append(np.mean(window_matrix, axis=0))

        reference = windows[-1]
        trajectories[group] = [_cosine_distance(window, reference) for window in windows]

    return trajectories


def compute_language_convergence(
    data: Sequence[Mapping[str, Any]] | Sequence[str],
    *,
    text_column: str = "text",
    group_column: str = "session_id",
    window_size: int = 3,
    slope_tolerance: float = 1e-6,
    model_name: str = "all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 32,
    device: str = "auto",
    embedder: Callable[[Sequence[str]], np.ndarray] | None = None,
    text_mapper: Callable[[Mapping[str, Any]], Any] | None = None,
) -> LanguageConvergenceResult:
    """Compute convergence/divergence of language trajectories by group.

    Negative slope indicates convergence toward the final language centroid.
    Positive slope indicates divergence.
    """
    trajectories = compute_semantic_distance_trajectory(
        data,
        text_column=text_column,
        group_column=group_column,
        window_size=window_size,
        model_name=model_name,
        normalize=normalize,
        batch_size=batch_size,
        device=device,
        embedder=embedder,
        text_mapper=text_mapper,
    )

    slope_by_group: dict[str, float] = {}
    direction_by_group: dict[str, str] = {}
    for group, distances in trajectories.items():
        if len(distances) < 2:
            slope = 0.0
        else:
            x = np.arange(len(distances), dtype=float)
            y = np.asarray(distances, dtype=float)
            slope = float(np.polyfit(x, y, deg=1)[0])

        if slope < -slope_tolerance:
            direction = "converging"
        elif slope > slope_tolerance:
            direction = "diverging"
        else:
            direction = "stable"

        slope_by_group[group] = slope
        direction_by_group[group] = direction

    n_observations = len(data) if _is_text_sequence(data) else len(coerce_unified_table(data))

    groups = sorted(trajectories)
    return LanguageConvergenceResult(
        groups=groups,
        distance_trajectories=trajectories,
        slope_by_group=slope_by_group,
        direction_by_group=direction_by_group,
        window_size=window_size,
        n_observations=n_observations,
        config={
            "text_column": text_column,
            "group_column": group_column,
            "window_size": int(window_size),
            "model_name": model_name if embedder is None else "custom",
            "normalized_embeddings": bool(normalize),
        },
    )


def _resolve_text_input(
    data: Sequence[Mapping[str, Any]] | Sequence[str],
    *,
    text_column: str,
) -> list[str]:
    if _is_text_sequence(data):
        return [str(item) for item in data]

    rows = coerce_unified_table(data)
    texts: list[str] = []
    for index, row in enumerate(rows):
        value = row.get(text_column)
        if _is_blank(value):
            raise ValueError(f"Row {index} is missing text in '{text_column}'.")
        texts.append(str(value))
    return texts


def fit_topic_model(
    data: Sequence[Mapping[str, Any]] | Sequence[str],
    *,
    n_topics: int = 5,
    max_features: int = 5000,
    random_state: int = 0,
    text_column: str = "text",
    top_k_terms: int = 10,
) -> dict[str, Any]:
    """Fit an LDA topic model and return topic summaries.

    Args:
        data: Unified table rows or a list of texts.
        n_topics: Number of latent topics.
        max_features: Maximum vocabulary size.
        random_state: Random seed.
        text_column: Text column for table input.
        top_k_terms: Number of representative terms per topic.

    Returns:
        JSON-serializable topic summary.
    """
    if n_topics <= 0:
        raise ValueError("n_topics must be positive.")
    if max_features <= 0:
        raise ValueError("max_features must be positive.")
    if top_k_terms <= 0:
        raise ValueError("top_k_terms must be positive.")

    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as exc:
        raise ImportError(_LANG_IMPORT_ERROR) from exc

    texts = _resolve_text_input(data, text_column=text_column)
    if not texts:
        raise ValueError("Topic modeling requires at least one text.")

    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
    )
    doc_topic = lda.fit_transform(matrix)

    terms = np.asarray(vectorizer.get_feature_names_out())
    topic_terms: list[dict[str, Any]] = []
    for topic_index, component in enumerate(lda.components_):
        order = np.argsort(component)[::-1][:top_k_terms]
        topic_terms.append(
            {
                "topic": int(topic_index),
                "terms": [str(terms[idx]) for idx in order],
                "weights": [float(component[idx]) for idx in order],
            }
        )

    return {
        "n_topics": int(n_topics),
        "n_documents": len(texts),
        "vocab_size": len(terms),
        "doc_topic_distribution": doc_topic.tolist(),
        "topic_terms": topic_terms,
        "config": {
            "max_features": int(max_features),
            "random_state": int(random_state),
            "text_column": text_column,
            "top_k_terms": int(top_k_terms),
        },
    }


def score_sentiment(
    data: Sequence[Mapping[str, Any]] | Sequence[str],
    *,
    text_column: str = "text",
) -> dict[str, Any]:
    """Score sentiment with a deterministic lexicon-based approach.

    This lightweight scorer is intentionally simple and offline-friendly.
    """
    texts = _resolve_text_input(data, text_column=text_column)

    scores: list[float] = []
    labels: list[str] = []
    for text in texts:
        tokens = [token.lower() for token in _TOKEN_PATTERN.findall(text)]
        if not tokens:
            score = 0.0
        else:
            pos = sum(1 for token in tokens if token in _POSITIVE_WORDS)
            neg = sum(1 for token in tokens if token in _NEGATIVE_WORDS)
            score = float(pos - neg) / float(len(tokens))

        if score > 0.03:
            label = "positive"
        elif score < -0.03:
            label = "negative"
        else:
            label = "neutral"

        scores.append(score)
        labels.append(label)

    scores_array = np.asarray(scores, dtype=float)
    return {
        "n_documents": len(texts),
        "scores": scores,
        "labels": labels,
        "mean_score": float(np.mean(scores_array)) if len(scores_array) else 0.0,
        "std_score": float(np.std(scores_array)) if len(scores_array) else 0.0,
        "counts": {
            "positive": int(sum(label == "positive" for label in labels)),
            "neutral": int(sum(label == "neutral" for label in labels)),
            "negative": int(sum(label == "negative" for label in labels)),
        },
        "config": {"text_column": text_column, "method": "lexicon"},
    }


__all__ = [
    "LanguageConvergenceResult",
    "compute_language_convergence",
    "compute_semantic_distance_trajectory",
    "fit_topic_model",
    "score_sentiment",
]
