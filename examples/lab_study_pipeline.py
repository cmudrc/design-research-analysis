"""Analyze a small prompt-framing lab study end-to-end.

## Introduction
Run a realistic small-sample lab workflow across control vs reframed conditions
with sequence, language, dataset, embedding-map, statistical, and
provenance outputs.

## Technical Implementation
1. Build an in-memory unified event table with condition labels and outcome fields.
2. Validate table quality and compute sequence/language summaries.
3. Profile and validate a dataframe schema; generate a codebook.
4. Run group comparison, regression, bootstrap, permutation, and power helpers.
5. Run PCA embedding-map clustering and write a reproducibility manifest with
   attached provenance payload.

## Expected Results
Prints concise summaries for state count, convergence labels, sentiment totals,
schema/profile diagnostics, key statistical metrics, clustering labels, and the
manifest path under ``artifacts/runtime``.

## References
- docs/workflows.rst
- docs/analysis_recipes.rst
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import design_research_analysis as dran


def main() -> None:
    """Run a compact, lab-authentic analysis workflow with reproducibility metadata."""
    rows = [
        {
            "timestamp": "2026-02-03T09:00:00Z",
            "session_id": "ctrl-01",
            "condition": "control",
            "actor_id": "designer-a",
            "event_type": "propose",
            "text": "good first concept from prior baseline",
            "novelty_score": 4.2,
            "cycle_time_min": 15.0,
        },
        {
            "timestamp": "2026-02-03T09:01:00Z",
            "session_id": "ctrl-01",
            "condition": "control",
            "actor_id": "designer-b",
            "event_type": "evaluate",
            "text": "difficult tradeoff discussion with weak evidence",
            "novelty_score": 4.0,
            "cycle_time_min": 17.0,
        },
        {
            "timestamp": "2026-02-03T09:02:00Z",
            "session_id": "ctrl-01",
            "condition": "control",
            "actor_id": "designer-a",
            "event_type": "refine",
            "text": "small improvement but still unclear mechanism",
            "novelty_score": 4.3,
            "cycle_time_min": 16.0,
        },
        {
            "timestamp": "2026-02-03T09:03:00Z",
            "session_id": "ctrl-01",
            "condition": "control",
            "actor_id": "designer-b",
            "event_type": "evaluate",
            "text": "better but risky integration path",
            "novelty_score": 4.1,
            "cycle_time_min": 16.5,
        },
        {
            "timestamp": "2026-02-03T10:00:00Z",
            "session_id": "reframe-01",
            "condition": "reframed",
            "actor_id": "designer-c",
            "event_type": "propose",
            "text": "clear reframed concept with strong rationale",
            "novelty_score": 6.3,
            "cycle_time_min": 12.0,
        },
        {
            "timestamp": "2026-02-03T10:01:00Z",
            "session_id": "reframe-01",
            "condition": "reframed",
            "actor_id": "designer-d",
            "event_type": "evaluate",
            "text": "helpful critique and collaborative option merge",
            "novelty_score": 6.6,
            "cycle_time_min": 11.5,
        },
        {
            "timestamp": "2026-02-03T10:02:00Z",
            "session_id": "reframe-01",
            "condition": "reframed",
            "actor_id": "designer-c",
            "event_type": "refine",
            "text": "effective refinement with successful constraint closure",
            "novelty_score": 6.8,
            "cycle_time_min": 10.5,
        },
        {
            "timestamp": "2026-02-03T10:03:00Z",
            "session_id": "reframe-01",
            "condition": "reframed",
            "actor_id": "designer-d",
            "event_type": "evaluate",
            "text": "excellent final concept and clear evidence trail",
            "novelty_score": 6.7,
            "cycle_time_min": 11.0,
        },
    ]

    table = dran.coerce_unified_table(rows, config=dran.UnifiedTableConfig())
    table = dran.derive_columns(table)
    report = dran.validate_unified_table(table)
    if not report.is_valid:
        raise RuntimeError(f"Unified table validation failed: {report.errors}")

    markov = dran.fit_markov_chain_from_table(table, order=1, smoothing=1.0)

    embedding_lookup = {
        row["text"]: [float(index), float(len(row["text"].split()))]
        for index, row in enumerate(table, start=1)
    }
    trajectory = dran.compute_semantic_distance_trajectory(
        table,
        window_size=2,
        embedder=lambda texts: [embedding_lookup[text] for text in texts],
    )
    convergence = dran.compute_language_convergence(
        table,
        window_size=2,
        embedder=lambda texts: [embedding_lookup[text] for text in texts],
    )
    sentiment = dran.score_sentiment(table)

    frame = pd.DataFrame(table)
    profile = dran.profile_dataframe(frame)
    schema_check = dran.validate_dataframe(
        frame,
        {
            "session_id": {"dtype": "string", "required": True, "nullable": False},
            "condition": {
                "dtype": "string",
                "required": True,
                "allowed": ["control", "reframed"],
            },
            "novelty_score": {"dtype": "numeric", "required": True, "min": 0.0, "max": 10.0},
            "cycle_time_min": {"dtype": "numeric", "required": True, "min": 0.0},
        },
    )
    codebook = dran.generate_codebook(frame)

    novelty = frame["novelty_score"].astype(float).tolist()
    cycle_time = frame["cycle_time_min"].astype(float).tolist()
    conditions = frame["condition"].astype(str).tolist()
    control = [score for score, cond in zip(novelty, conditions, strict=True) if cond == "control"]
    reframed = [
        score for score, cond in zip(novelty, conditions, strict=True) if cond == "reframed"
    ]

    group_test = dran.compare_groups(values=novelty, groups=conditions, method="ttest")
    regression = dran.fit_regression(
        [[minutes] for minutes in cycle_time],
        novelty,
        feature_names=["cycle_time_min"],
    )
    bootstrap = dran.bootstrap_ci(novelty, n_resamples=500, seed=11)
    permutation = dran.permutation_test(control, reframed, n_permutations=500, seed=11)
    sample_size = dran.estimate_sample_size(
        effect_size=0.8,
        test="two_sample_t",
        alpha=0.05,
        power=0.8,
    )
    curve = dran.power_curve([0.2, 0.5, 0.8], n=24, test="two_sample_t")
    mde = dran.minimum_detectable_effect(n=24, test="two_sample_t", alpha=0.05, power=0.8)

    vectors = np.asarray(
        [
            [4.2, 15.0, 1.0],
            [4.1, 16.8, 1.2],
            [6.6, 11.2, 2.2],
            [6.7, 10.8, 2.4],
        ],
        dtype=float,
    )
    embedding_map = dran.build_embedding_map(vectors, method="pca", n_components=2)
    clusters = dran.cluster_embedding_map(embedding_map, method="kmeans", n_clusters=2)

    context = dran.capture_run_context(seed=11)
    manifest_path = Path("artifacts/runtime/lab_study_manifest.json")
    dran.write_run_manifest(context, manifest_path)
    payload = dran.attach_provenance(
        {
            "table_ok": report.is_valid,
            "schema_ok": bool(schema_check["ok"]),
            "codebook_columns": len(codebook),
        },
        context,
    )

    print(f"Markov states: {len(markov.states)}")
    print(f"Trajectory groups: {sorted(trajectory)}")
    print(f"Convergence labels: {convergence.direction_by_group}")
    print(f"Sentiment docs: {sentiment['n_documents']}")
    print(f"Profiled rows: {profile['n_rows']}")
    print(f"Group p-value: {group_test.p_value:.3f}")
    print(f"Regression R2: {regression.r2:.3f}")
    print(f"Bootstrap estimate: {bootstrap['estimate']:.3f}")
    print(f"Permutation p-value: {permutation['p_value']:.3f}")
    print(f"Recommended n: {sample_size['recommended_n']}")
    print(f"Power curve points: {len(curve)}")
    print(f"MDE at n=24: {mde['minimum_detectable_effect']:.3f}")
    print(f"Cluster labels: {clusters['labels']}")
    print(f"Manifest written: {manifest_path}")
    print(f"Payload keys: {sorted(payload)}")


if __name__ == "__main__":
    main()
