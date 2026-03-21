"""Mechanical design-review analysis example.

## Introduction
Analyze a compact mechanical design review centered on a lightweight mounting
bracket redesign. The example keeps the workflow small while surfacing the
sequence, statistics, embedding, and reporting helpers that are useful in
engineering design studies.

## Technical Implementation
1. Validate and summarize a tiny unified event table from two bracket-review sessions.
2. Fit and compare Markov-chain traces, build a condition metric table, and run
   a permutation-style pair comparison on mass-oriented scores.
3. Embed the review notes with a deterministic custom embedder and render quick
   transition visualizations for the session traces.

## Expected Results
The script prints validation status, state labels, one significance summary, and
an embedding shape for the bracket-design review sessions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import design_research_analysis as dran


def _review_rows() -> list[dict[str, object]]:
    """Return one tiny bracket-design review table."""
    return [
        {
            "timestamp": "2026-02-01T09:00:00Z",
            "session_id": "bracket-baseline",
            "actor_id": "analyst-a",
            "event_type": "propose",
            "text": "baseline bracket keeps the rib layout but stays heavy near the mounting boss",
        },
        {
            "timestamp": "2026-02-01T09:02:00Z",
            "session_id": "bracket-baseline",
            "actor_id": "analyst-b",
            "event_type": "evaluate",
            "text": "stress stays acceptable yet the mounting boss still drives unnecessary mass",
        },
        {
            "timestamp": "2026-02-01T09:04:00Z",
            "session_id": "bracket-baseline",
            "actor_id": "analyst-a",
            "event_type": "refine",
            "text": "reduce web thickness and taper the fillet to remove aluminum from the boss",
        },
        {
            "timestamp": "2026-02-01T09:06:00Z",
            "session_id": "bracket-lightweight",
            "actor_id": "analyst-a",
            "event_type": "propose",
            "text": "lightweight concept adds a diagonal rib and shortens the unsupported flange",
        },
        {
            "timestamp": "2026-02-01T09:08:00Z",
            "session_id": "bracket-lightweight",
            "actor_id": "analyst-b",
            "event_type": "evaluate",
            "text": (
                "the diagonal rib improves stiffness and should reduce peak "
                "deflection at the bolt hole"
            ),
        },
        {
            "timestamp": "2026-02-01T09:10:00Z",
            "session_id": "bracket-lightweight",
            "actor_id": "analyst-a",
            "event_type": "refine",
            "text": (
                "final concept trims flange mass while preserving load path "
                "continuity into the base plate"
            ),
        },
    ]


def _custom_embedder(texts: list[str]) -> np.ndarray:
    """Return one deterministic embedding matrix for example text."""
    vectors = []
    for index, text in enumerate(texts, start=1):
        token_count = len(text.split())
        vectors.append([float(index), float(token_count), float(len(set(text.split())))])
    return np.asarray(vectors, dtype=float)


def _follow_on_analysis_toolkit() -> dict[str, object]:
    """Return advanced follow-on tools relevant to design-review studies."""
    return {
        "comparison_result_type": dran.ComparisonResult,
        "decode_result_type": dran.DecodeResult,
        "discrete_hmm_result_type": dran.DiscreteHMMResult,
        "gaussian_hmm_result_type": dran.GaussianHMMResult,
        "markov_result_type": dran.MarkovChainResult,
        "table_report_type": dran.UnifiedTableValidationReport,
        "dataset_module": dran.dataset,
        "embedding_maps_module": dran.embedding_maps,
        "embedding_result_type": dran.EmbeddingResult,
        "language_module": dran.language,
        "runtime_module": dran.runtime,
        "sequence_module": dran.sequence,
        "stats_module": dran.stats,
        "visualization_module": dran.visualization,
        "decode_hmm": dran.decode_hmm,
        "embed_records": dran.embed_records,
        "fit_discrete_hmm_from_table": dran.fit_discrete_hmm_from_table,
        "fit_mixed_effects": dran.fit_mixed_effects,
        "fit_text_gaussian_hmm_from_table": dran.fit_text_gaussian_hmm_from_table,
        "fit_topic_model": dran.fit_topic_model,
        "is_google_colab": dran.is_google_colab,
        "is_notebook": dran.is_notebook,
        "plot_state_graph": dran.plot_state_graph,
        "plot_transition_matrix": dran.plot_transition_matrix,
        "rank_tests_one_stop": dran.rank_tests_one_stop,
    }


def main() -> None:
    """Run a lightweight mechanical design-review analysis workflow."""
    rows = _review_rows()
    report = dran.validate_unified_table(rows)
    if not report.is_valid:
        raise RuntimeError(f"Invalid unified table: {report.errors}")

    markov = dran.fit_markov_chain_from_table(rows, order=1, smoothing=1.0)
    alternate = dran.fit_markov_chain_from_table(list(reversed(rows)), order=1, smoothing=1.0)
    comparison = markov - alternate
    print("Validation:", report.is_valid, "warnings", len(report.warnings))
    print("States:", ", ".join(str(state) for state in markov.states))
    print("Comparison object:", isinstance(comparison, dran.ComparisonResult))

    condition_rows = [
        {"condition_id": "baseline", "condition": "baseline"},
        {"condition_id": "lightweight", "condition": "lightweight"},
    ]
    run_rows = [
        {"run_id": "run-1", "condition_id": "baseline"},
        {"run_id": "run-2", "condition_id": "baseline"},
        {"run_id": "run-3", "condition_id": "lightweight"},
        {"run_id": "run-4", "condition_id": "lightweight"},
    ]
    evaluation_rows = [
        {
            "run_id": "run-1",
            "metric_name": "mass_score",
            "metric_value": 0.68,
            "aggregation_level": "run",
        },
        {
            "run_id": "run-2",
            "metric_name": "mass_score",
            "metric_value": 0.71,
            "aggregation_level": "run",
        },
        {
            "run_id": "run-3",
            "metric_name": "mass_score",
            "metric_value": 0.81,
            "aggregation_level": "run",
        },
        {
            "run_id": "run-4",
            "metric_name": "mass_score",
            "metric_value": 0.84,
            "aggregation_level": "run",
        },
    ]
    condition_metric_table = dran.build_condition_metric_table(
        run_rows,
        metric="mass_score",
        evaluations=evaluation_rows,
        conditions=condition_rows,
    )
    pair_report = dran.compare_condition_pairs(condition_metric_table, metric_name="mass_score")
    print(pair_report.render_brief().splitlines()[1])

    rank_result = dran.rank_tests_one_stop([0.68, 0.71], [0.81, 0.84], kind="mannwhitney")
    print("Rank test:", rank_result["test"], f"p={rank_result['p_value']:.4f}")

    embedding_result = dran.embed_records(
        rows,
        record_id_mapper=lambda row: f"{row['session_id']}-{row['event_type']}",
        embedder=_custom_embedder,
    )
    print("Embedding shape:", embedding_result.embeddings.shape)
    print("Embedding result object:", isinstance(embedding_result, dran.EmbeddingResult))

    topic_summary = dran.fit_topic_model(
        [str(row["text"]) for row in rows], n_topics=2, top_k_terms=3
    )
    print("Topic model topics:", topic_summary["n_topics"])

    transition_figure, _ = dran.plot_transition_matrix(
        markov, title="Bracket Review Transition Matrix"
    )
    state_graph_figure, _ = dran.plot_state_graph(
        markov,
        title="Bracket Review State Graph",
        threshold=0.0,
    )
    plt.close(transition_figure)
    plt.close(state_graph_figure)

    toolkit = _follow_on_analysis_toolkit()
    print(
        "Toolkit modules:",
        ", ".join(
            module.__name__.split(".")[-1]
            for module in (
                toolkit["dataset_module"],
                toolkit["embedding_maps_module"],
                toolkit["language_module"],
                toolkit["runtime_module"],
                toolkit["sequence_module"],
                toolkit["stats_module"],
                toolkit["visualization_module"],
            )
        ),
    )
    print("Follow-on tools tracked:", len(toolkit))


if __name__ == "__main__":
    main()
