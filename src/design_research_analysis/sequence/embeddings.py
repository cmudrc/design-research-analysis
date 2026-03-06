"""Embedding helpers for sequence-model pipelines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_EMBED_IMPORT_ERROR = (
    "Text embedding requires optional embedding dependencies. "
    "Install with `pip install design-research-analysis[embeddings]`."
)


def embed_text(
    texts: Sequence[str],
    *,
    model_name: str = "all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 32,
    device: str = "auto",
) -> np.ndarray:
    """Encode text into dense vectors using SentenceTransformers.

    Args:
        texts: Ordered text observations.
        model_name: SentenceTransformers model identifier.
        normalize: Whether to L2-normalize vectors.
        batch_size: Embedding batch size.
        device: Runtime device name such as ``cpu``, ``cuda``, or ``auto``.

    Returns:
        Array of shape ``(n_texts, embedding_dim)``.

    Raises:
        ImportError: If optional embedding dependencies are unavailable.
        ValueError: If input is empty or malformed.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    items = [str(item) for item in texts]
    if not items:
        raise ValueError("texts must contain at least one item.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(_EMBED_IMPORT_ERROR) from exc

    model_device = None if device == "auto" else device
    model = SentenceTransformer(model_name, device=model_device)
    embedded = model.encode(
        items,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    arr = np.asarray(embedded, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Embedded output must be a 2D matrix.")
    return arr
