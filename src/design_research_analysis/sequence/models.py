"""Model-training helpers for Markov chains and Hidden Markov Models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._backend import get_hmm_backend
from .embeddings import embed_text

Token = Hashable


def _serialize_token(token: Token) -> str | int | float | bool | None:
    """Convert tokens to JSON-friendly values when possible."""
    if token is None or isinstance(token, (str, int, float, bool)):
        return token
    return repr(token)


def _as_float_matrix(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array-like structure.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    return arr


def _validate_lengths(lengths: Sequence[int] | None, n_samples: int) -> list[int] | None:
    if lengths is None:
        return None
    normalized = [int(item) for item in lengths]
    if not normalized:
        raise ValueError("lengths must not be empty.")
    if any(item <= 0 for item in normalized):
        raise ValueError("All lengths must be positive.")
    if sum(normalized) != n_samples:
        raise ValueError("Sum of lengths must equal number of observations.")
    return normalized


def _normalize_token_sequences(
    token_sequences: Sequence[Token] | Sequence[Sequence[Token]],
) -> list[list[Token]]:
    if len(token_sequences) == 0:
        raise ValueError("token_sequences must not be empty.")

    first = token_sequences[0]
    if isinstance(first, (str, bytes)) or not isinstance(first, Sequence):
        return [[item for item in token_sequences]]

    normalized: list[list[Token]] = []
    for seq in token_sequences:
        if isinstance(seq, (str, bytes)) or not isinstance(seq, Sequence):
            raise ValueError("token_sequences must contain either tokens or token sequences.")
        tokens = [item for item in seq]
        if not tokens:
            raise ValueError("All token sequences must contain at least one token.")
        normalized.append(tokens)
    return normalized


def _flatten_token_sequences(
    token_sequences: Sequence[Token] | Sequence[Sequence[Token]],
) -> tuple[list[Token], list[int]]:
    sequences = _normalize_token_sequences(token_sequences)
    flat: list[Token] = []
    lengths: list[int] = []
    for seq in sequences:
        flat.extend(seq)
        lengths.append(len(seq))
    return flat, lengths


def _normalize_text_observations(
    texts: Sequence[str] | Sequence[Sequence[str]],
) -> tuple[list[str], list[int] | None]:
    if len(texts) == 0:
        raise ValueError("texts must not be empty.")

    first = texts[0]
    if isinstance(first, str):
        items = [str(item) for item in texts]
        return items, None

    if not isinstance(first, Sequence):
        raise ValueError("texts must be a sequence of strings or a sequence of string sequences.")

    flat_items: list[str] = []
    lengths: list[int] = []
    for sequence in texts:
        seq = [str(item) for item in sequence]
        if not seq:
            raise ValueError("Nested text sequences must not be empty.")
        flat_items.extend(seq)
        lengths.append(len(seq))
    return flat_items, lengths


def _transition_like_matrix(model_result: Any) -> np.ndarray:
    if isinstance(model_result, MarkovChainResult):
        return model_result.transition_matrix
    if isinstance(model_result, (GaussianHMMResult, DiscreteHMMResult)):
        return model_result.transmat
    raise TypeError("Expected MarkovChainResult, GaussianHMMResult, or DiscreteHMMResult.")


def _state_labels(model_result: Any) -> list[str]:
    if isinstance(model_result, MarkovChainResult):
        if model_result.order == 1:
            return [str(state[0]) for state in model_result.states]
        return ["|".join(str(part) for part in state) for state in model_result.states]

    if isinstance(model_result, (GaussianHMMResult, DiscreteHMMResult)):
        return [f"S{idx}" for idx in range(model_result.n_states)]

    raise TypeError("Expected MarkovChainResult, GaussianHMMResult, or DiscreteHMMResult.")


@dataclass(slots=True)
class MarkovChainResult:
    """Serializable result container for an order-k Markov chain."""

    order: int
    states: list[tuple[Token, ...]]
    transition_matrix: np.ndarray
    startprob: np.ndarray
    smoothing: float
    n_sequences: int
    n_observations: int
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "order": int(self.order),
            "states": [[_serialize_token(token) for token in state] for state in self.states],
            "transition_matrix": self.transition_matrix.tolist(),
            "startprob": self.startprob.tolist(),
            "smoothing": float(self.smoothing),
            "n_sequences": int(self.n_sequences),
            "n_observations": int(self.n_observations),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class GaussianHMMResult:
    """Serializable result container for a Gaussian HMM."""

    model: Any = field(repr=False)
    backend: str = "hmmlearn"
    n_states: int = 0
    covariance_type: str = "diag"
    seed: int = 0
    lengths: list[int] | None = None
    startprob: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    transmat: np.ndarray = field(default_factory=lambda: np.array([[]], dtype=float))
    means: np.ndarray = field(default_factory=lambda: np.array([[]], dtype=float))
    covars: np.ndarray = field(default_factory=lambda: np.array([[]], dtype=float))
    train_log_likelihood: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "backend": self.backend,
            "n_states": int(self.n_states),
            "covariance_type": self.covariance_type,
            "seed": int(self.seed),
            "lengths": list(self.lengths) if self.lengths is not None else None,
            "startprob": self.startprob.tolist(),
            "transmat": self.transmat.tolist(),
            "means": self.means.tolist(),
            "covars": self.covars.tolist(),
            "train_log_likelihood": float(self.train_log_likelihood),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class DiscreteHMMResult:
    """Serializable result container for a discrete-emission HMM."""

    model: Any = field(repr=False)
    backend: str = "hmmlearn"
    n_states: int = 0
    seed: int = 0
    lengths: list[int] | None = None
    startprob: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    transmat: np.ndarray = field(default_factory=lambda: np.array([[]], dtype=float))
    emissionprob: np.ndarray = field(default_factory=lambda: np.array([[]], dtype=float))
    vocab: list[Token] = field(default_factory=list)
    token_to_id: dict[Token, int] = field(default_factory=dict, repr=False)
    train_log_likelihood: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "backend": self.backend,
            "n_states": int(self.n_states),
            "seed": int(self.seed),
            "lengths": list(self.lengths) if self.lengths is not None else None,
            "startprob": self.startprob.tolist(),
            "transmat": self.transmat.tolist(),
            "emissionprob": self.emissionprob.tolist(),
            "vocab": [_serialize_token(token) for token in self.vocab],
            "train_log_likelihood": float(self.train_log_likelihood),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class DecodeResult:
    """Serializable decoded-state output from an HMM."""

    algorithm: str
    log_probability: float
    states: np.ndarray
    lengths: list[int] | None
    backend: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the decode output to a JSON-serializable dictionary."""
        return {
            "algorithm": self.algorithm,
            "log_probability": float(self.log_probability),
            "states": self.states.astype(int).tolist(),
            "lengths": list(self.lengths) if self.lengths is not None else None,
            "backend": self.backend,
        }


def fit_markov_chain(
    token_sequences: Sequence[Token] | Sequence[Sequence[Token]],
    *,
    order: int = 1,
    smoothing: float = 1.0,
) -> MarkovChainResult:
    """Fit an order-k Markov chain from token sequences.

    Args:
        token_sequences: A token sequence or a list of token sequences.
        order: Markov order (number of previous symbols in each state).
        smoothing: Additive smoothing applied to transition and start counts.

    Returns:
        A :class:`MarkovChainResult` with normalized transition probabilities.
    """
    if order <= 0:
        raise ValueError("order must be positive.")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")

    sequences = _normalize_token_sequences(token_sequences)

    start_counts: Counter[tuple[Token, ...]] = Counter()
    transition_counts: dict[tuple[Token, ...], Counter[tuple[Token, ...]]] = {}
    states: set[tuple[Token, ...]] = set()

    total_transitions = 0
    for seq in sequences:
        if len(seq) < order:
            continue

        start_state = tuple(seq[:order])
        start_counts[start_state] += 1
        states.add(start_state)

        if len(seq) == order:
            continue

        for idx in range(order, len(seq)):
            src = tuple(seq[idx - order : idx])
            dst = tuple(seq[idx - order + 1 : idx + 1])
            transition_counts.setdefault(src, Counter())[dst] += 1
            states.add(src)
            states.add(dst)
            total_transitions += 1

    if not states or total_transitions == 0:
        raise ValueError(
            "Not enough sequence length to estimate transitions. "
            "Provide sequences with at least order + 1 observations."
        )

    ordered_states = sorted(states, key=lambda state: tuple(str(part) for part in state))
    state_to_index = {state: idx for idx, state in enumerate(ordered_states)}
    n_states = len(ordered_states)

    transition = np.full((n_states, n_states), smoothing, dtype=float)
    for src, counts in transition_counts.items():
        src_idx = state_to_index[src]
        for dst, count in counts.items():
            dst_idx = state_to_index[dst]
            transition[src_idx, dst_idx] += float(count)

    for row_idx in range(n_states):
        row_sum = float(transition[row_idx].sum())
        if row_sum <= 0:
            transition[row_idx] = 1.0 / n_states
        else:
            transition[row_idx] /= row_sum

    startprob = np.full(n_states, smoothing, dtype=float)
    for state, count in start_counts.items():
        startprob[state_to_index[state]] += float(count)
    start_total = float(startprob.sum())
    if start_total <= 0:
        startprob[:] = 1.0 / n_states
    else:
        startprob /= start_total

    return MarkovChainResult(
        order=order,
        states=ordered_states,
        transition_matrix=transition,
        startprob=startprob,
        smoothing=smoothing,
        n_sequences=len(sequences),
        n_observations=int(sum(len(seq) for seq in sequences)),
        config={"order": order, "smoothing": smoothing},
    )


def fit_gaussian_hmm(
    X: Any,
    *,
    lengths: Sequence[int] | None = None,
    n_states: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 100,
    seed: int = 0,
    backend: str = "hmmlearn",
) -> GaussianHMMResult:
    """Fit a Gaussian-emission Hidden Markov Model.

    Args:
        X: Observation matrix with shape ``(n_samples, n_features)``.
        lengths: Optional sequence lengths for multiple trajectories.
        n_states: Number of hidden states.
        covariance_type: Covariance model passed to ``hmmlearn.GaussianHMM``.
        n_iter: Maximum EM iterations.
        seed: Random seed.
        backend: HMM backend name.

    Returns:
        Fitted Gaussian HMM result.
    """
    if n_states <= 0:
        raise ValueError("n_states must be positive.")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")

    obs = _as_float_matrix(X, name="X")
    seq_lengths = _validate_lengths(lengths, obs.shape[0])

    adapter = get_hmm_backend(backend)
    model = adapter.create_gaussian_hmm(
        n_states=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        seed=seed,
    )
    model.fit(obs, lengths=seq_lengths)

    train_ll = float(model.score(obs, lengths=seq_lengths))
    return GaussianHMMResult(
        model=model,
        backend=adapter.name,
        n_states=int(n_states),
        covariance_type=covariance_type,
        seed=int(seed),
        lengths=seq_lengths,
        startprob=np.asarray(model.startprob_, dtype=float).copy(),
        transmat=np.asarray(model.transmat_, dtype=float).copy(),
        means=np.asarray(model.means_, dtype=float).copy(),
        covars=np.asarray(model.covars_, dtype=float).copy(),
        train_log_likelihood=train_ll,
        config={
            "n_states": int(n_states),
            "covariance_type": covariance_type,
            "n_iter": int(n_iter),
            "seed": int(seed),
            "backend": adapter.name,
        },
    )


def fit_text_gaussian_hmm(
    texts: Sequence[str] | Sequence[Sequence[str]],
    *,
    n_states: int = 3,
    embedder: Callable[[Sequence[str]], np.ndarray] | None = None,
    lengths: Sequence[int] | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 32,
    device: str = "auto",
    covariance_type: str = "diag",
    n_iter: int = 100,
    seed: int = 0,
    backend: str = "hmmlearn",
) -> GaussianHMMResult:
    """Embed text observations and fit a Gaussian HMM over embeddings.

    Args:
        texts: Flat text observations or grouped text sequences.
        n_states: Number of hidden states.
        embedder: Optional custom callable that maps texts to an embedding matrix.
        lengths: Optional sequence lengths; inferred from grouped inputs when omitted.
        model_name: SentenceTransformers model name used when ``embedder`` is omitted.
        normalize: Whether to normalize embeddings when using built-in embedding.
        batch_size: Embedding batch size.
        device: Embedding device, for example ``cpu`` or ``cuda``.
        covariance_type: Gaussian covariance type.
        n_iter: Maximum EM iterations.
        seed: Random seed.
        backend: HMM backend name.

    Returns:
        Fitted Gaussian HMM result with embedding metadata in ``config``.
    """
    flat_texts, inferred_lengths = _normalize_text_observations(texts)
    resolved_lengths = lengths if lengths is not None else inferred_lengths

    if embedder is None:
        embeddings = embed_text(
            flat_texts,
            model_name=model_name,
            normalize=normalize,
            batch_size=batch_size,
            device=device,
        )
        embedding_config = {
            "embedding_provider": "sentence-transformers",
            "embedding_model": model_name,
            "embedding_normalized": bool(normalize),
            "embedding_batch_size": int(batch_size),
            "embedding_device": device,
        }
    else:
        embeddings = np.asarray(embedder(flat_texts), dtype=float)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)
        if embeddings.ndim != 2:
            raise ValueError("embedder must return a 2D embedding matrix.")
        embedding_config = {
            "embedding_provider": "callable",
            "embedding_model": "custom",
        }

    result = fit_gaussian_hmm(
        embeddings,
        lengths=resolved_lengths,
        n_states=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        seed=seed,
        backend=backend,
    )
    result.config.update(embedding_config)
    return result


def fit_discrete_hmm(
    token_sequences: Sequence[Token] | Sequence[Sequence[Token]],
    *,
    n_states: int = 3,
    n_iter: int = 100,
    seed: int = 0,
    backend: str = "hmmlearn",
) -> DiscreteHMMResult:
    """Fit a discrete-emission HMM from token sequences.

    Args:
        token_sequences: A token sequence or list of token sequences.
        n_states: Number of hidden states.
        n_iter: Maximum EM iterations.
        seed: Random seed.
        backend: HMM backend name.

    Returns:
        Fitted discrete HMM result.
    """
    if n_states <= 0:
        raise ValueError("n_states must be positive.")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")

    flat_tokens, seq_lengths = _flatten_token_sequences(token_sequences)
    if not flat_tokens:
        raise ValueError("token_sequences must contain at least one token.")

    token_to_id: dict[Token, int] = {}
    encoded: list[int] = []
    for token in flat_tokens:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
        encoded.append(token_to_id[token])

    vocab_by_id = {idx: token for token, idx in token_to_id.items()}
    vocab = [vocab_by_id[idx] for idx in range(len(vocab_by_id))]

    X = np.asarray(encoded, dtype=int).reshape(-1, 1)

    adapter = get_hmm_backend(backend)
    model = adapter.create_discrete_hmm(
        n_states=n_states,
        n_iter=n_iter,
        seed=seed,
        n_symbols=len(vocab),
    )
    model.fit(X, lengths=seq_lengths)

    train_ll = float(model.score(X, lengths=seq_lengths))
    return DiscreteHMMResult(
        model=model,
        backend=adapter.name,
        n_states=int(n_states),
        seed=int(seed),
        lengths=list(seq_lengths),
        startprob=np.asarray(model.startprob_, dtype=float).copy(),
        transmat=np.asarray(model.transmat_, dtype=float).copy(),
        emissionprob=np.asarray(model.emissionprob_, dtype=float).copy(),
        vocab=vocab,
        token_to_id=token_to_id,
        train_log_likelihood=train_ll,
        config={
            "n_states": int(n_states),
            "n_iter": int(n_iter),
            "seed": int(seed),
            "n_symbols": int(len(vocab)),
            "backend": adapter.name,
        },
    )


def decode_hmm(
    model_result: GaussianHMMResult | DiscreteHMMResult,
    observations: Any,
    *,
    algorithm: str = "viterbi",
    lengths: Sequence[int] | None = None,
) -> DecodeResult:
    """Decode the most likely hidden-state sequence for observations.

    Args:
        model_result: Fitted Gaussian or discrete HMM result object.
        observations: Observation matrix (Gaussian) or token sequences (discrete).
        algorithm: Decoding algorithm, ``viterbi`` or ``map``.
        lengths: Optional sequence lengths for batched observations.

    Returns:
        Decoded state sequence and log probability.
    """
    if algorithm not in {"viterbi", "map"}:
        raise ValueError("algorithm must be one of: viterbi, map")

    if isinstance(model_result, GaussianHMMResult):
        obs = _as_float_matrix(observations, name="observations")
        seq_lengths = _validate_lengths(lengths, obs.shape[0])
        log_prob, states = model_result.model.decode(obs, lengths=seq_lengths, algorithm=algorithm)
        return DecodeResult(
            algorithm=algorithm,
            log_probability=float(log_prob),
            states=np.asarray(states, dtype=int),
            lengths=seq_lengths,
            backend=model_result.backend,
        )

    if isinstance(model_result, DiscreteHMMResult):
        flat_tokens, inferred_lengths = _flatten_token_sequences(observations)
        seq_lengths_raw: Sequence[int] | None = lengths if lengths is not None else inferred_lengths

        encoded: list[int] = []
        for token in flat_tokens:
            if token not in model_result.token_to_id:
                raise ValueError(f"Observation token {token!r} is not in the fitted vocabulary.")
            encoded.append(model_result.token_to_id[token])

        X = np.asarray(encoded, dtype=int).reshape(-1, 1)
        normalized_lengths = _validate_lengths(seq_lengths_raw, X.shape[0])
        log_prob, states = model_result.model.decode(
            X,
            lengths=normalized_lengths,
            algorithm=algorithm,
        )
        return DecodeResult(
            algorithm=algorithm,
            log_probability=float(log_prob),
            states=np.asarray(states, dtype=int),
            lengths=normalized_lengths,
            backend=model_result.backend,
        )

    raise TypeError("model_result must be GaussianHMMResult or DiscreteHMMResult")


__all__ = [
    "DecodeResult",
    "DiscreteHMMResult",
    "GaussianHMMResult",
    "MarkovChainResult",
    "decode_hmm",
    "fit_discrete_hmm",
    "fit_gaussian_hmm",
    "fit_markov_chain",
    "fit_text_gaussian_hmm",
    "_state_labels",
    "_transition_like_matrix",
]
