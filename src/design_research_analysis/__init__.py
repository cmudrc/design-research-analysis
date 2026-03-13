"""Curated public exports for design-research-analysis."""

from . import dataset, dimred, language, runtime, sequence, stats
from ._comparison import ComparisonResult
from .dataset import generate_codebook, profile_dataframe, validate_dataframe
from .dimred import cluster_projection, embed_records, reduce_dimensions
from .language import (
    compute_language_convergence,
    compute_semantic_distance_trajectory,
    fit_topic_model,
    score_sentiment,
)
from .runtime import (
    attach_provenance,
    capture_run_context,
    is_google_colab,
    is_notebook,
    write_run_manifest,
)
from .sequence import (
    DecodeResult,
    DiscreteHMMResult,
    GaussianHMMResult,
    MarkovChainResult,
    decode_hmm,
    fit_discrete_hmm_from_table,
    fit_markov_chain_from_table,
    fit_text_gaussian_hmm_from_table,
    plot_state_graph,
    plot_transition_matrix,
)
from .stats import (
    bootstrap_ci,
    compare_groups,
    estimate_sample_size,
    fit_mixed_effects,
    fit_regression,
    minimum_detectable_effect,
    permutation_test,
    power_curve,
    rank_tests_one_stop,
)
from .table import (
    UnifiedTableConfig,
    UnifiedTableValidationReport,
    coerce_unified_table,
    derive_columns,
    validate_unified_table,
)

__all__ = [
    "ComparisonResult",
    "DecodeResult",
    "DiscreteHMMResult",
    "GaussianHMMResult",
    "MarkovChainResult",
    "UnifiedTableConfig",
    "UnifiedTableValidationReport",
    "attach_provenance",
    "bootstrap_ci",
    "capture_run_context",
    "cluster_projection",
    "coerce_unified_table",
    "compare_groups",
    "compute_language_convergence",
    "compute_semantic_distance_trajectory",
    "dataset",
    "decode_hmm",
    "derive_columns",
    "dimred",
    "embed_records",
    "estimate_sample_size",
    "fit_discrete_hmm_from_table",
    "fit_markov_chain_from_table",
    "fit_mixed_effects",
    "fit_regression",
    "fit_text_gaussian_hmm_from_table",
    "fit_topic_model",
    "generate_codebook",
    "is_google_colab",
    "is_notebook",
    "language",
    "minimum_detectable_effect",
    "permutation_test",
    "plot_state_graph",
    "plot_transition_matrix",
    "power_curve",
    "profile_dataframe",
    "rank_tests_one_stop",
    "reduce_dimensions",
    "runtime",
    "score_sentiment",
    "sequence",
    "stats",
    "validate_dataframe",
    "validate_unified_table",
    "write_run_manifest",
]
