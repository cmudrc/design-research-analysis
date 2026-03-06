"""Curated public exports for design-research-analysis."""

from . import dimred, language, sequence, stats
from .dimred import cluster_projection, embed_records, reduce_dimensions
from .language import (
    compute_language_convergence,
    compute_semantic_distance_trajectory,
    fit_topic_model,
    score_sentiment,
)
from .sequence import (
    fit_discrete_hmm_from_table,
    fit_markov_chain_from_table,
    fit_text_gaussian_hmm_from_table,
    plot_state_graph,
    plot_transition_matrix,
)
from .stats import compare_groups, fit_mixed_effects, fit_regression
from .table import (
    UnifiedTableConfig,
    UnifiedTableValidationReport,
    coerce_unified_table,
    derive_columns,
    validate_unified_table,
)

__all__ = [
    "UnifiedTableConfig",
    "UnifiedTableValidationReport",
    "cluster_projection",
    "coerce_unified_table",
    "compare_groups",
    "compute_language_convergence",
    "compute_semantic_distance_trajectory",
    "derive_columns",
    "dimred",
    "embed_records",
    "fit_discrete_hmm_from_table",
    "fit_markov_chain_from_table",
    "fit_mixed_effects",
    "fit_regression",
    "fit_text_gaussian_hmm_from_table",
    "fit_topic_model",
    "language",
    "plot_state_graph",
    "plot_transition_matrix",
    "reduce_dimensions",
    "score_sentiment",
    "sequence",
    "stats",
    "validate_unified_table",
]
