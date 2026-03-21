from __future__ import annotations

import pytest

from design_research_analysis import build_condition_metric_table, compare_condition_pairs


def test_build_condition_metric_table_joins_conditions_for_run_metric() -> None:
    runs = [
        {"run_id": "run-1", "condition_id": "cond-a", "primary_outcome": 0.61},
        {"run_id": "run-2", "condition_id": "cond-b", "primary_outcome": 0.44},
    ]
    conditions = [
        {"condition_id": "cond-a", "selection_strategy": "neutral_prompt"},
        {"condition_id": "cond-b", "selection_strategy": "random_selection"},
    ]

    table = build_condition_metric_table(
        runs,
        metric="primary_outcome",
        condition_column="selection_strategy",
        conditions=conditions,
    )

    assert table == [
        {
            "run_id": "run-1",
            "condition_id": "cond-a",
            "condition": "neutral_prompt",
            "metric": "primary_outcome",
            "value": 0.61,
            "condition_source": "conditions",
            "metric_source": "runs",
        },
        {
            "run_id": "run-2",
            "condition_id": "cond-b",
            "condition": "random_selection",
            "metric": "primary_outcome",
            "value": 0.44,
            "condition_source": "conditions",
            "metric_source": "runs",
        },
    ]


def test_build_condition_metric_table_reads_evaluations_metric() -> None:
    runs = [
        {"run_id": "run-1", "condition_id": "cond-a"},
        {"run_id": "run-2", "condition_id": "cond-b"},
    ]
    conditions = [
        {"condition_id": "cond-a", "selection_strategy": "neutral_prompt"},
        {"condition_id": "cond-b", "selection_strategy": "random_selection"},
    ]
    evaluations = [
        {
            "run_id": "run-1",
            "metric_name": "market_share_proxy",
            "metric_value": 0.73,
            "aggregation_level": "run",
        },
        {
            "run_id": "run-2",
            "metric_name": "market_share_proxy",
            "metric_value": 0.41,
            "aggregation_level": "run",
        },
        {
            "run_id": "run-1",
            "metric_name": "market_share_proxy",
            "metric_value": 999.0,
            "aggregation_level": "record",
        },
    ]

    table = build_condition_metric_table(
        runs,
        metric="market_share_proxy",
        condition_column="selection_strategy",
        conditions=conditions,
        evaluations=evaluations,
    )

    assert [row["value"] for row in table] == [0.73, 0.41]
    assert {row["metric_source"] for row in table} == {"evaluations"}


def test_build_condition_metric_table_errors_on_duplicate_condition_definitions() -> None:
    runs = [{"run_id": "run-1", "condition_id": "cond-a", "primary_outcome": 0.5}]
    conditions = [
        {"condition_id": "cond-a", "selection_strategy": "neutral_prompt"},
        {"condition_id": "cond-a", "selection_strategy": "profit_focus_prompt"},
    ]

    with pytest.raises(ValueError, match="duplicate 'condition_id' value"):
        build_condition_metric_table(
            runs,
            metric="primary_outcome",
            condition_column="selection_strategy",
            conditions=conditions,
        )


def test_build_condition_metric_table_errors_on_missing_metric() -> None:
    runs = [{"run_id": "run-1", "condition_id": "cond-a"}]
    conditions = [{"condition_id": "cond-a", "selection_strategy": "neutral_prompt"}]
    evaluations = [{"run_id": "run-1", "metric_name": "other_metric", "metric_value": 0.5}]

    with pytest.raises(ValueError, match="No evaluation metric 'market_share_proxy'"):
        build_condition_metric_table(
            runs,
            metric="market_share_proxy",
            condition_column="selection_strategy",
            conditions=conditions,
            evaluations=evaluations,
        )


def test_build_condition_metric_table_errors_on_duplicate_evaluation_metric_rows() -> None:
    runs = [{"run_id": "run-1", "condition_id": "cond-a"}]
    conditions = [{"condition_id": "cond-a", "selection_strategy": "neutral_prompt"}]
    evaluations = [
        {"run_id": "run-1", "metric_name": "market_share_proxy", "metric_value": 0.6},
        {"run_id": "run-1", "metric_name": "market_share_proxy", "metric_value": 0.7},
    ]

    with pytest.raises(ValueError, match="Multiple evaluation rows matched metric"):
        build_condition_metric_table(
            runs,
            metric="market_share_proxy",
            condition_column="selection_strategy",
            conditions=conditions,
            evaluations=evaluations,
        )


def test_compare_condition_pairs_exact_path_and_significance_rows() -> None:
    joined = [
        {"condition": "neutral_prompt", "metric": "market_share_proxy", "value": 0.61},
        {"condition": "neutral_prompt", "metric": "market_share_proxy", "value": 0.63},
        {"condition": "random_selection", "metric": "market_share_proxy", "value": 0.40},
        {"condition": "random_selection", "metric": "market_share_proxy", "value": 0.42},
    ]

    report = compare_condition_pairs(
        joined,
        condition_pairs=[("neutral_prompt", "random_selection")],
        alternative="greater",
        alpha=0.2,
    )

    assert report.metric == "market_share_proxy"
    assert len(report.comparisons) == 1

    comparison = report.comparisons[0]
    assert comparison.test_method == "exact"
    assert comparison.permutations_evaluated == 6
    assert comparison.total_permutations == 6
    assert comparison.p_value == pytest.approx(1.0 / 6.0)
    assert comparison.mean_difference > 0.0
    assert comparison.effect_size > 0.0
    assert comparison.higher_condition == "neutral_prompt"
    assert comparison.significant is True

    rows = report.to_significance_rows()
    assert len(rows) == 1
    assert rows[0]["test"] == "exact_permutation_test"
    assert rows[0]["outcome"] == "market_share_proxy (neutral_prompt vs random_selection)"
    assert rows[0]["p_value"] == pytest.approx(1.0 / 6.0)
    assert rows[0]["effect_size"] == pytest.approx(comparison.effect_size)
    assert rows[0]["mean_difference"] == pytest.approx(comparison.mean_difference)
    assert rows[0]["higher_condition"] == "neutral_prompt"
    assert rows[0]["alternative"] == "greater"
    assert rows[0]["test_method"] == "exact"
    assert rows[0]["permutations_evaluated"] == 6
    assert rows[0]["total_permutations"] == 6
    assert rows[0]["significant"] is True
    assert "neutral_prompt vs random_selection" in report.render_brief()


def test_compare_condition_pairs_sampled_fallback_and_pair_ordering() -> None:
    joined = [
        {"condition": "B", "metric": "score", "value": 1.1},
        {"condition": "A", "metric": "score", "value": 0.1},
        {"condition": "C", "metric": "score", "value": 2.1},
        {"condition": "B", "metric": "score", "value": 1.0},
        {"condition": "A", "metric": "score", "value": 0.2},
        {"condition": "C", "metric": "score", "value": 2.0},
    ]

    report = compare_condition_pairs(
        joined,
        exact_threshold=1,
        n_permutations=25,
        seed=3,
    )

    assert [(item.left_condition, item.right_condition) for item in report.comparisons] == [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    ]
    assert all(item.test_method == "sampled" for item in report.comparisons)

    right_heavier = report.comparisons[-1]
    assert right_heavier.mean_difference < 0.0
    assert right_heavier.effect_size < 0.0
    assert right_heavier.higher_condition == "C"


def test_compare_condition_pairs_validation_errors() -> None:
    with pytest.raises(ValueError, match="At least two conditions"):
        compare_condition_pairs([{"condition": "A", "value": 1.0}])

    with pytest.raises(ValueError, match="No condition pairs were requested"):
        compare_condition_pairs(
            [
                {"condition": "A", "value": 1.0},
                {"condition": "B", "value": 2.0},
            ],
            condition_pairs=[],
        )

    with pytest.raises(ValueError, match="multiple metric labels"):
        compare_condition_pairs(
            [
                {"condition": "A", "metric": "score_a", "value": 1.0},
                {"condition": "B", "metric": "score_b", "value": 2.0},
            ]
        )
