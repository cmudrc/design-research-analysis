"""Sequence modeling utilities for Markov chains and Hidden Markov Models."""

from .embeddings import embed_text
from .models import (
    DecodeResult,
    DiscreteHMMResult,
    GaussianHMMResult,
    MarkovChainResult,
    decode_hmm,
    fit_discrete_hmm,
    fit_discrete_hmm_from_table,
    fit_gaussian_hmm,
    fit_markov_chain,
    fit_markov_chain_from_table,
    fit_text_gaussian_hmm,
    fit_text_gaussian_hmm_from_table,
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
    "fit_discrete_hmm_from_table",
    "fit_gaussian_hmm",
    "fit_markov_chain",
    "fit_markov_chain_from_table",
    "fit_text_gaussian_hmm",
    "fit_text_gaussian_hmm_from_table",
    "plot_state_graph",
    "plot_transition_matrix",
]
