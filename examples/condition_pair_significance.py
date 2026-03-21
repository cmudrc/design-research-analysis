"""Compare experiment-exported condition metrics with pairwise significance tests.

## Introduction
Turn canonical ``runs.csv`` / ``conditions.csv`` / ``evaluations.csv`` style
rows into a joined condition-metric table, then compute pairwise permutation
tests and effect sizes without hand-rolled analysis glue.

## Technical Implementation
1. Define in-memory canonical export rows for runs, conditions, and evaluations.
2. Build a normalized run-level table for ``market_share_proxy`` by joining
   condition labels onto evaluation metrics.
3. Compare ordered condition pairs and print a concise brief plus one
   significance row shaped for ``design_research_experiments.render_significance_brief``.

## Expected Results
Prints the normalized joined-row count, a markdown-ready condition comparison
brief, and the first structured significance row.

## References
- docs/experiments_handoff.rst
- docs/analysis_recipes.rst
"""

from __future__ import annotations

import design_research_analysis as dran


def main() -> None:
    """Run a compact condition-comparison workflow over canonical export rows."""
    runs = [
        {"run_id": "run-1", "condition_id": "cond-random"},
        {"run_id": "run-2", "condition_id": "cond-random"},
        {"run_id": "run-3", "condition_id": "cond-random"},
        {"run_id": "run-4", "condition_id": "cond-neutral"},
        {"run_id": "run-5", "condition_id": "cond-neutral"},
        {"run_id": "run-6", "condition_id": "cond-neutral"},
        {"run_id": "run-7", "condition_id": "cond-profit"},
        {"run_id": "run-8", "condition_id": "cond-profit"},
        {"run_id": "run-9", "condition_id": "cond-profit"},
    ]

    conditions = [
        {"condition_id": "cond-random", "selection_strategy": "random_selection"},
        {"condition_id": "cond-neutral", "selection_strategy": "neutral_prompt"},
        {"condition_id": "cond-profit", "selection_strategy": "profit_focus_prompt"},
    ]

    evaluations = [
        {"run_id": "run-1", "metric_name": "market_share_proxy", "metric_value": 0.40},
        {"run_id": "run-2", "metric_name": "market_share_proxy", "metric_value": 0.43},
        {"run_id": "run-3", "metric_name": "market_share_proxy", "metric_value": 0.41},
        {"run_id": "run-4", "metric_name": "market_share_proxy", "metric_value": 0.57},
        {"run_id": "run-5", "metric_name": "market_share_proxy", "metric_value": 0.59},
        {"run_id": "run-6", "metric_name": "market_share_proxy", "metric_value": 0.60},
        {"run_id": "run-7", "metric_name": "market_share_proxy", "metric_value": 0.69},
        {"run_id": "run-8", "metric_name": "market_share_proxy", "metric_value": 0.72},
        {"run_id": "run-9", "metric_name": "market_share_proxy", "metric_value": 0.71},
    ]

    joined = dran.build_condition_metric_table(
        runs,
        metric="market_share_proxy",
        condition_column="selection_strategy",
        conditions=conditions,
        evaluations=evaluations,
    )
    report = dran.compare_condition_pairs(
        joined,
        condition_pairs=[
            ("neutral_prompt", "random_selection"),
            ("profit_focus_prompt", "neutral_prompt"),
            ("profit_focus_prompt", "random_selection"),
        ],
        alternative="greater",
        seed=17,
    )

    print(f"Joined rows: {len(joined)}")
    print(report.render_brief())
    print(report.to_significance_rows()[0])


if __name__ == "__main__":
    main()
