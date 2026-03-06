"""Public package exports for design-research-analysis."""

from . import sequence
from .core import (
    ProjectBlueprint,
    build_default_blueprint,
    describe_project,
    normalize_package_name,
)
from .sequence import (
    DecodeResult,
    DiscreteHMMResult,
    GaussianHMMResult,
    MarkovChainResult,
    decode_hmm,
    embed_text,
    fit_discrete_hmm,
    fit_gaussian_hmm,
    fit_markov_chain,
    fit_text_gaussian_hmm,
    plot_state_graph,
    plot_transition_matrix,
)

__all__ = [
    "DecodeResult",
    "DiscreteHMMResult",
    "GaussianHMMResult",
    "MarkovChainResult",
    "ProjectBlueprint",
    "build_default_blueprint",
    "decode_hmm",
    "describe_project",
    "embed_text",
    "fit_discrete_hmm",
    "fit_gaussian_hmm",
    "fit_markov_chain",
    "fit_text_gaussian_hmm",
    "normalize_package_name",
    "plot_state_graph",
    "plot_transition_matrix",
    "sequence",
]
