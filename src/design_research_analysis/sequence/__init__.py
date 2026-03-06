"""Sequence modeling utilities for Markov chains and Hidden Markov Models."""

from .embeddings import embed_text
from .models import (
    DecodeResult,
    DiscreteHMMResult,
    GaussianHMMResult,
    MarkovChainResult,
    decode_hmm,
    fit_discrete_hmm,
    fit_gaussian_hmm,
    fit_markov_chain,
    fit_text_gaussian_hmm,
)
from .visualization import plot_state_graph, plot_transition_matrix

__all__ = [
    "DecodeResult",
    "DiscreteHMMResult",
    "GaussianHMMResult",
    "MarkovChainResult",
    "decode_hmm",
    "embed_text",
    "fit_discrete_hmm",
    "fit_gaussian_hmm",
    "fit_markov_chain",
    "fit_text_gaussian_hmm",
    "plot_state_graph",
    "plot_transition_matrix",
]
