"""Model-training helpers for Markov chains and Hidden Markov Models."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._comparison import (
    ComparableResultMixin,
    ComparisonResult,
    align_square_matrix_by_labels,
    align_vector_by_labels,
    best_assignment,
    permute_square_matrix,
    permute_vector,
)
from ..table import UnifiedTableConfig, coerce_unified_table, derive_columns, validate_unified_table
from ._backend import get_hmm_backend
from .embeddings import embed_text

Token = Hashable


def _serialize_token(token: Token) -> str | int | float | bool | None:
    """Convert tokens to JSON-friendly values when possible."""
    if token is None or isinstance(token, (str, int, float, bool)):
        return token
    return repr(token)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


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


def _chi_square_statistic(contingency: np.ndarray) -> tuple[float, int]:
    if contingency.ndim != 2:
        raise ValueError("Chi-square comparison requires a 2D contingency table.")
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0, 0

    total = float(contingency.sum())
    if total <= 0.0:
        return 0.0, 0

    row_sums = contingency.sum(axis=1, keepdims=True)
    col_sums = contingency.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / total
    mask = expected > 0.0
    chi2 = float(np.sum(((contingency - expected) ** 2)[mask] / expected[mask]))
    dof = int((contingency.shape[0] - 1) * (contingency.shape[1] - 1))
    return chi2, dof


def _chi_square_p_value(statistic: float, dof: int) -> float | None:
    if dof <= 0:
        return None
    try:
        from scipy.stats import chi2
    except ImportError:
        return None
    return float(chi2.sf(statistic, dof))


@dataclass(slots=True)
class MarkovChainResult(ComparableResultMixin):
    """Serializable result container for an order-k Markov chain."""

    order: int
    states: list[tuple[Token, ...]]
    transition_matrix: np.ndarray
    startprob: np.ndarray
    smoothing: float
    n_sequences: int
    n_observations: int
    config: dict[str, Any] = field(default_factory=dict)
    _transition_counts: np.ndarray = field(
        default_factory=lambda: np.array([[]], dtype=float),
        repr=False,
    )
    _start_counts: np.ndarray = field(default_factory=lambda: np.array([], dtype=float), repr=False)

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

    def _comparison_metric(self) -> str:
        return "transition_profile"

    def _aligned_comparison_payload(self, other: MarkovChainResult) -> dict[str, Any]:
        if self.order != other.order:
            raise ValueError("Markov-chain comparison requires the same order on both results.")

        left_labels = _state_labels(self)
        right_labels = _state_labels(other)
        labels = sorted(set(left_labels) | set(right_labels))

        left_start = align_vector_by_labels(self.startprob, left_labels, labels)
        right_start = align_vector_by_labels(other.startprob, right_labels, labels)
        left_transition = align_square_matrix_by_labels(self.transition_matrix, left_labels, labels)
        right_transition = align_square_matrix_by_labels(
            other.transition_matrix, right_labels, labels
        )
        left_start_counts = align_vector_by_labels(self._start_counts, left_labels, labels)
        right_start_counts = align_vector_by_labels(other._start_counts, right_labels, labels)
        left_transition_counts = align_square_matrix_by_labels(
            self._transition_counts,
            left_labels,
            labels,
        )
        right_transition_counts = align_square_matrix_by_labels(
            other._transition_counts,
            right_labels,
            labels,
        )

        return {
            "state_labels": labels,
            "left_start": left_start,
            "right_start": right_start,
            "left_transition": left_transition,
            "right_transition": right_transition,
            "left_start_counts": left_start_counts,
            "right_start_counts": right_start_counts,
            "left_transition_counts": left_transition_counts,
            "right_transition_counts": right_transition_counts,
        }

    def _comparison_vectors(
        self,
        other: MarkovChainResult,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        payload = self._aligned_comparison_payload(other)
        left_vector = np.concatenate(
            [payload["left_start"], payload["left_transition"].reshape(-1)],
        )
        right_vector = np.concatenate(
            [payload["right_start"], payload["right_transition"].reshape(-1)],
        )
        return (
            left_vector,
            right_vector,
            {
                "state_labels": payload["state_labels"],
                "startprob_delta": (payload["left_start"] - payload["right_start"])
                .astype(float)
                .tolist(),
                "transition_delta": (payload["left_transition"] - payload["right_transition"])
                .astype(float)
                .tolist(),
            },
        )

    def _build_comparison(self, other: Any, *, operation: str) -> ComparisonResult:
        payload = self._aligned_comparison_payload(other)
        left_transition = payload["left_transition"]
        right_transition = payload["right_transition"]
        start_delta = payload["left_start"] - payload["right_start"]
        transition_delta = left_transition - right_transition

        left_profile_counts = np.concatenate(
            [payload["left_start_counts"], payload["left_transition_counts"].reshape(-1)],
        )
        right_profile_counts = np.concatenate(
            [payload["right_start_counts"], payload["right_transition_counts"].reshape(-1)],
        )
        contingency = np.vstack([left_profile_counts, right_profile_counts])
        contingency = contingency[:, contingency.sum(axis=0) > 0.0]
        chi2, dof = _chi_square_statistic(contingency)
        total = float(contingency.sum())
        scale = min(contingency.shape[0] - 1, contingency.shape[1] - 1) if contingency.size else 0
        effect = float(math.sqrt(chi2 / (total * scale))) if total > 0.0 and scale > 0 else 0.0
        p_value = _chi_square_p_value(chi2, dof)
        estimate = float(np.linalg.norm(transition_delta))
        details = {
            "state_labels": payload["state_labels"],
            "transition_delta": transition_delta.astype(float).tolist(),
            "startprob_delta": start_delta.astype(float).tolist(),
            "degrees_of_freedom": int(dof),
            "n_profile_cells": int(contingency.shape[1]) if contingency.ndim == 2 else 0,
        }

        if operation == "difference":
            if p_value is None:
                interpretation = (
                    f"Aligned transition-profile difference has Frobenius norm {estimate:.4g}. "
                    f"Chi-square statistic is {chi2:.4g} with df={dof}. "
                    f"Effect size V={effect:.4g}. Install scipy for a p-value."
                )
            else:
                interpretation = (
                    f"Aligned transition-profile difference has Frobenius norm {estimate:.4g}. "
                    f"Chi-square statistic is {chi2:.4g} with df={dof} and p={p_value:.4g}. "
                    f"Effect size V={effect:.4g}."
                )
            return ComparisonResult(
                operation="difference",
                left_type=type(self).__name__,
                right_type=type(other).__name__,
                metric=self._comparison_metric(),
                estimate=estimate,
                statistic=float(chi2),
                p_value=p_value,
                effect_size=effect,
                details=details,
                interpretation=interpretation,
            )

        if operation == "effect_size":
            interpretation = (
                f"Transition-profile effect size is V={effect:.4g}. "
                "Larger values indicate stronger differences in start or transition structure."
            )
            return ComparisonResult(
                operation="effect_size",
                left_type=type(self).__name__,
                right_type=type(other).__name__,
                metric=self._comparison_metric(),
                estimate=effect,
                statistic=float(chi2),
                p_value=p_value,
                effect_size=effect,
                details=details,
                interpretation=interpretation,
            )

        raise ValueError(f"Unsupported comparison operation: {operation}")


@dataclass(slots=True)
class GaussianHMMResult(ComparableResultMixin):
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

    def _comparison_metric(self) -> str:
        return "hidden_state_profile"

    def _comparison_vectors(
        self,
        other: GaussianHMMResult,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self.n_states != other.n_states:
            raise ValueError("Gaussian-HMM comparison requires the same number of states.")
        if self.means.shape != other.means.shape:
            raise ValueError("Gaussian-HMM comparison requires matching mean matrix shapes.")
        if (
            np.asarray(self.covars, dtype=float).shape
            != np.asarray(other.covars, dtype=float).shape
        ):
            raise ValueError("Gaussian-HMM comparison requires matching covariance shapes.")

        cost = np.linalg.norm(self.means[:, None, :] - other.means[None, :, :], axis=2)
        permutation = best_assignment(cost)

        right_start = permute_vector(other.startprob, permutation)
        right_transition = permute_square_matrix(other.transmat, permutation)
        right_means = np.asarray(other.means, dtype=float)[list(permutation), ...]
        right_covars = np.asarray(other.covars, dtype=float)[list(permutation), ...]

        left_vector = np.concatenate(
            [
                np.asarray(self.startprob, dtype=float).reshape(-1),
                np.asarray(self.transmat, dtype=float).reshape(-1),
                np.asarray(self.means, dtype=float).reshape(-1),
                np.asarray(self.covars, dtype=float).reshape(-1),
            ]
        )
        right_vector = np.concatenate(
            [
                np.asarray(right_start, dtype=float).reshape(-1),
                np.asarray(right_transition, dtype=float).reshape(-1),
                np.asarray(right_means, dtype=float).reshape(-1),
                np.asarray(right_covars, dtype=float).reshape(-1),
            ]
        )
        return (
            left_vector,
            right_vector,
            {
                "state_permutation": list(permutation),
                "startprob_delta": (
                    np.asarray(self.startprob, dtype=float) - np.asarray(right_start, dtype=float)
                ).tolist(),
                "transmat_delta": (
                    np.asarray(self.transmat, dtype=float)
                    - np.asarray(right_transition, dtype=float)
                ).tolist(),
                "means_delta": (
                    np.asarray(self.means, dtype=float) - np.asarray(right_means, dtype=float)
                ).tolist(),
                "covars_delta": (
                    np.asarray(self.covars, dtype=float) - np.asarray(right_covars, dtype=float)
                ).tolist(),
                "train_log_likelihood_delta": float(
                    self.train_log_likelihood - other.train_log_likelihood
                ),
            },
        )


@dataclass(slots=True)
class DiscreteHMMResult(ComparableResultMixin):
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

    def _comparison_metric(self) -> str:
        return "hidden_state_profile"

    def _comparison_vectors(
        self,
        other: DiscreteHMMResult,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self.n_states != other.n_states:
            raise ValueError("Discrete-HMM comparison requires the same number of states.")

        vocab_labels = sorted(
            {repr(_serialize_token(token)) for token in self.vocab}
            | {repr(_serialize_token(token)) for token in other.vocab}
        )
        left_vocab_labels = [repr(_serialize_token(token)) for token in self.vocab]
        right_vocab_labels = [repr(_serialize_token(token)) for token in other.vocab]
        left_emission = np.vstack(
            [
                align_vector_by_labels(row, left_vocab_labels, vocab_labels)
                for row in self.emissionprob
            ]
        )
        right_emission = np.vstack(
            [
                align_vector_by_labels(row, right_vocab_labels, vocab_labels)
                for row in other.emissionprob
            ]
        )

        cost = np.linalg.norm(left_emission[:, None, :] - right_emission[None, :, :], axis=2)
        permutation = best_assignment(cost)
        right_start = permute_vector(other.startprob, permutation)
        right_transition = permute_square_matrix(other.transmat, permutation)
        right_emission_aligned = right_emission[list(permutation), :]

        left_vector = np.concatenate(
            [
                np.asarray(self.startprob, dtype=float).reshape(-1),
                np.asarray(self.transmat, dtype=float).reshape(-1),
                np.asarray(left_emission, dtype=float).reshape(-1),
            ]
        )
        right_vector = np.concatenate(
            [
                np.asarray(right_start, dtype=float).reshape(-1),
                np.asarray(right_transition, dtype=float).reshape(-1),
                np.asarray(right_emission_aligned, dtype=float).reshape(-1),
            ]
        )
        return (
            left_vector,
            right_vector,
            {
                "state_permutation": list(permutation),
                "aligned_vocab": list(vocab_labels),
                "startprob_delta": (
                    np.asarray(self.startprob, dtype=float) - np.asarray(right_start, dtype=float)
                ).tolist(),
                "transmat_delta": (
                    np.asarray(self.transmat, dtype=float)
                    - np.asarray(right_transition, dtype=float)
                ).tolist(),
                "emissionprob_delta": (
                    np.asarray(left_emission, dtype=float)
                    - np.asarray(right_emission_aligned, dtype=float)
                ).tolist(),
                "train_log_likelihood_delta": float(
                    self.train_log_likelihood - other.train_log_likelihood
                ),
            },
        )


@dataclass(slots=True)
class DecodeResult(ComparableResultMixin):
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

    def _comparison_metric(self) -> str:
        return "decoded_state_profile"

    def _comparison_vectors(
        self,
        other: DecodeResult,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        left_vector = np.concatenate(
            [
                self.states.astype(float).reshape(-1),
                np.asarray([self.log_probability], dtype=float),
            ]
        )
        right_vector = np.concatenate(
            [
                other.states.astype(float).reshape(-1),
                np.asarray([other.log_probability], dtype=float),
            ]
        )
        return (
            left_vector,
            right_vector,
            {
                "algorithms": [self.algorithm, other.algorithm],
                "backends": [self.backend, other.backend],
                "lengths": [self.lengths, other.lengths],
            },
        )


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

    transition_count_matrix = np.zeros((n_states, n_states), dtype=float)
    for src, counts in transition_counts.items():
        src_idx = state_to_index[src]
        for dst, count in counts.items():
            dst_idx = state_to_index[dst]
            transition_count_matrix[src_idx, dst_idx] = float(count)

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

    start_counts_vector = np.zeros(n_states, dtype=float)
    for state, count in start_counts.items():
        start_counts_vector[state_to_index[state]] = float(count)

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
        _transition_counts=transition_count_matrix,
        _start_counts=start_counts_vector,
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
            "n_symbols": len(vocab),
            "backend": adapter.name,
        },
    )


def _prepare_table_rows(
    table: Sequence[Mapping[str, Any]],
    *,
    config: UnifiedTableConfig | None,
    actor_mapper: Callable[[Mapping[str, Any]], Any] | None,
    event_mapper: Callable[[Mapping[str, Any]], Any] | None,
    session_mapper: Callable[[Mapping[str, Any]], Any] | None,
    text_mapper: Callable[[Mapping[str, Any]], Any] | None,
) -> list[dict[str, Any]]:
    rows = coerce_unified_table(table, config=config)
    rows = derive_columns(
        rows,
        actor_mapper=actor_mapper,
        event_mapper=event_mapper,
        session_mapper=session_mapper,
        text_mapper=text_mapper,
    )
    report = validate_unified_table(rows, config=config)
    if not report.is_valid:
        raise ValueError("Invalid unified table: " + "; ".join(report.errors))
    return rows


def _extract_grouped_tokens(
    rows: Sequence[Mapping[str, Any]],
    *,
    event_column: str,
    session_column: str,
    actor_column: str,
    include_actor_in_token: bool,
) -> list[list[Token]]:
    grouped: dict[str, list[Token]] = {}
    for index, row in enumerate(rows):
        event = row.get(event_column)
        if _is_blank(event):
            raise ValueError(
                f"Row {index} is missing '{event_column}'. Provide event values or an event mapper."
            )

        session_value = row.get(session_column)
        session = "__all__" if _is_blank(session_value) else str(session_value)

        token: Token
        if include_actor_in_token:
            actor = row.get(actor_column)
            if _is_blank(actor):
                raise ValueError(
                    f"Row {index} is missing '{actor_column}' while include_actor_in_token=True."
                )
            token = (str(actor), event)
        else:
            token = event

        grouped.setdefault(session, []).append(token)

    sequences = [grouped[key] for key in sorted(grouped)]
    non_empty = [seq for seq in sequences if len(seq) > 0]
    if not non_empty:
        raise ValueError("No valid token sequences were produced from the table.")
    return non_empty


def fit_markov_chain_from_table(
    table: Sequence[Mapping[str, Any]],
    *,
    order: int = 1,
    smoothing: float = 1.0,
    event_column: str = "event_type",
    session_column: str = "session_id",
    actor_column: str = "actor_id",
    include_actor_in_token: bool = False,
    actor_mapper: Callable[[Mapping[str, Any]], Any] | None = None,
    event_mapper: Callable[[Mapping[str, Any]], Any] | None = None,
    session_mapper: Callable[[Mapping[str, Any]], Any] | None = None,
    table_config: UnifiedTableConfig | None = None,
) -> MarkovChainResult:
    """Fit a Markov chain from unified-table event records."""
    rows = _prepare_table_rows(
        table,
        config=table_config,
        actor_mapper=actor_mapper,
        event_mapper=event_mapper,
        session_mapper=session_mapper,
        text_mapper=None,
    )
    sequences = _extract_grouped_tokens(
        rows,
        event_column=event_column,
        session_column=session_column,
        actor_column=actor_column,
        include_actor_in_token=include_actor_in_token,
    )

    result = fit_markov_chain(sequences, order=order, smoothing=smoothing)
    result.config.update(
        {
            "source": "table",
            "event_column": event_column,
            "session_column": session_column,
            "actor_column": actor_column,
            "include_actor_in_token": bool(include_actor_in_token),
        }
    )
    return result


def fit_discrete_hmm_from_table(
    table: Sequence[Mapping[str, Any]],
    *,
    n_states: int = 3,
    n_iter: int = 100,
    seed: int = 0,
    backend: str = "hmmlearn",
    event_column: str = "event_type",
    session_column: str = "session_id",
    actor_column: str = "actor_id",
    include_actor_in_token: bool = False,
    actor_mapper: Callable[[Mapping[str, Any]], Any] | None = None,
    event_mapper: Callable[[Mapping[str, Any]], Any] | None = None,
    session_mapper: Callable[[Mapping[str, Any]], Any] | None = None,
    table_config: UnifiedTableConfig | None = None,
) -> DiscreteHMMResult:
    """Fit a discrete HMM from unified-table event records."""
    rows = _prepare_table_rows(
        table,
        config=table_config,
        actor_mapper=actor_mapper,
        event_mapper=event_mapper,
        session_mapper=session_mapper,
        text_mapper=None,
    )
    sequences = _extract_grouped_tokens(
        rows,
        event_column=event_column,
        session_column=session_column,
        actor_column=actor_column,
        include_actor_in_token=include_actor_in_token,
    )

    result = fit_discrete_hmm(
        sequences,
        n_states=n_states,
        n_iter=n_iter,
        seed=seed,
        backend=backend,
    )
    result.config.update(
        {
            "source": "table",
            "event_column": event_column,
            "session_column": session_column,
            "actor_column": actor_column,
            "include_actor_in_token": bool(include_actor_in_token),
        }
    )
    return result


def fit_text_gaussian_hmm_from_table(
    table: Sequence[Mapping[str, Any]],
    *,
    text_column: str = "text",
    session_column: str = "session_id",
    n_states: int = 3,
    embedder: Callable[[Sequence[str]], np.ndarray] | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 32,
    device: str = "auto",
    covariance_type: str = "diag",
    n_iter: int = 100,
    seed: int = 0,
    backend: str = "hmmlearn",
    session_mapper: Callable[[Mapping[str, Any]], Any] | None = None,
    text_mapper: Callable[[Mapping[str, Any]], Any] | None = None,
    table_config: UnifiedTableConfig | None = None,
) -> GaussianHMMResult:
    """Embed text from unified-table rows and fit a Gaussian HMM."""
    rows = _prepare_table_rows(
        table,
        config=table_config,
        actor_mapper=None,
        event_mapper=None,
        session_mapper=session_mapper,
        text_mapper=text_mapper,
    )

    grouped: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        text = row.get(text_column)
        if _is_blank(text):
            raise ValueError(
                f"Row {index} is missing '{text_column}'. Provide text values or a text mapper."
            )
        session_value = row.get(session_column)
        session = "__all__" if _is_blank(session_value) else str(session_value)
        grouped.setdefault(session, []).append(str(text))

    if not grouped:
        raise ValueError("No text observations were found for Gaussian HMM fitting.")

    ordered_sessions = sorted(grouped)
    sequences = [grouped[key] for key in ordered_sessions]
    lengths = [len(seq) for seq in sequences] if len(ordered_sessions) > 1 else None
    flat_texts = [text for seq in sequences for text in seq]

    result = fit_text_gaussian_hmm(
        flat_texts,
        n_states=n_states,
        embedder=embedder,
        lengths=lengths,
        model_name=model_name,
        normalize=normalize,
        batch_size=batch_size,
        device=device,
        covariance_type=covariance_type,
        n_iter=n_iter,
        seed=seed,
        backend=backend,
    )
    result.config.update(
        {
            "source": "table",
            "text_column": text_column,
            "session_column": session_column,
            "n_sessions": len(ordered_sessions),
        }
    )
    return result


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
    "_state_labels",
    "_transition_like_matrix",
    "decode_hmm",
    "fit_discrete_hmm",
    "fit_discrete_hmm_from_table",
    "fit_gaussian_hmm",
    "fit_markov_chain",
    "fit_markov_chain_from_table",
    "fit_text_gaussian_hmm",
    "fit_text_gaussian_hmm_from_table",
]
