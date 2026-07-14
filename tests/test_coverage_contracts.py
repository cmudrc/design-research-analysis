"""Focused contracts for statistical, sequence, and CLI boundary behavior."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import design_research_analysis.cli as cli
import design_research_analysis.sequence.models as sequence_models
import design_research_analysis.stats as stats
from design_research_analysis.embedding_maps import EmbeddingMapResult


def test_group_comparison_result_serialization_and_algebra() -> None:
    """Group summaries should serialize and compare across partially overlapping labels."""
    left = stats.GroupComparisonResult(
        method="ttest",
        statistic=2.0,
        p_value=0.04,
        effect_size=0.8,
        group_means={"a": 1.0, "b": 2.0},
        group_sizes={"a": 3, "b": 4},
        config={"equal_var": False},
    )
    right = stats.GroupComparisonResult(
        method="kruskal",
        statistic=1.0,
        p_value=0.2,
        effect_size=0.2,
        group_means={"b": 1.5, "c": 3.0},
        group_sizes={"b": 5, "c": 2},
    )

    assert left.to_dict()["config"] == {"equal_var": False}
    difference = left - right
    assert difference.metric == "group_summary"
    assert difference.details["group_labels"] == ["a", "b", "c"]


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="scipy unavailable")
def test_compare_groups_supports_table_anova_kruskal_and_validation() -> None:
    """The table adapter should support each documented multi-group method."""
    rows = [
        {"score": 1.0, "team": "a"},
        {"score": 1.5, "team": "a"},
        {"score": 2.0, "team": "b"},
        {"score": 2.5, "team": "b"},
        {"score": 4.0, "team": "c"},
        {"score": 4.5, "team": "c"},
    ]

    anova = stats.compare_groups(data=rows, value_column="score", group_column="team")
    kruskal = stats.compare_groups(
        data=rows,
        value_column="score",
        group_column="team",
        method="kruskal",
    )
    assert anova.method == "anova"
    assert kruskal.method == "kruskal"
    assert anova.effect_size > 0.0

    with pytest.raises(ValueError, match="ttest requires exactly two"):
        stats.compare_groups(data=rows, value_column="score", group_column="team", method="ttest")
    with pytest.raises(ValueError, match="Unsupported method"):
        stats.compare_groups([1.0, 2.0], ["a", "b"], method="unknown")
    with pytest.raises(ValueError, match="missing 'score' or 'team'"):
        stats.compare_groups(data=[{"score": 1.0}], value_column="score", group_column="team")


def test_empty_condition_report_is_explicit() -> None:
    """Empty comparison reports should remain serializable and readable."""
    report = stats.ConditionComparisonReport(
        metric="score",
        condition_column="condition",
        metric_column="value",
        alternative="two-sided",
        alpha=0.05,
        comparisons=(),
    )
    assert report.to_dict()["comparisons"] == []
    assert report.to_significance_rows() == []
    assert "No condition comparisons" in report.render_brief()


def test_mixed_effects_result_and_sampled_pair_are_serializable() -> None:
    """Statistical result containers should preserve optional fields and labels."""
    left = stats.MixedEffectsResult(
        success=True,
        backend="statsmodels",
        formula="score ~ condition",
        group_column="participant",
        params={"condition": 0.5},
        aic=10.0,
        bic=12.0,
        log_likelihood=-3.0,
    )
    right = stats.MixedEffectsResult(
        success=False,
        backend="statsmodels",
        formula="score ~ 1",
        group_column="team",
        params={"intercept": 1.0},
        aic=None,
        bic=None,
        log_likelihood=None,
        message="did not converge",
    )
    assert left.to_dict()["aic"] == 10.0
    assert (left - right).details["parameter_names"] == ["condition", "intercept"]

    pair = stats.ConditionPairComparison(
        metric="score",
        left_condition="a",
        right_condition="b",
        mean_left=1.0,
        mean_right=2.0,
        n_left=3,
        n_right=3,
        mean_difference=-1.0,
        effect_size=-0.5,
        p_value=0.1,
        alternative="less",
        test_method="sampled",
        permutations_evaluated=100,
        total_permutations=200,
        higher_condition="b",
        significant=False,
    )
    assert pair.test_name == "sampled_permutation_test"
    assert pair.to_dict()["pair_label"] == "a vs b"


def test_statistical_scalar_helpers_cover_degenerate_samples() -> None:
    """Small and constant samples should have deterministic zero-effect behavior."""
    assert stats._cohen_d(np.asarray([1.0]), np.asarray([2.0])) == 0.0
    assert stats._cohen_d(np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0])) == 0.0
    assert stats._eta_squared([np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0])]) == 0.0
    with pytest.raises(ValueError, match="at least one value"):
        stats._as_array([], "values")
    assert stats._calc_stat(np.asarray([1.0, 3.0]), np.asarray([0.0, 2.0]), "diff_medians") == 1.0
    with pytest.raises(ValueError, match="must be one of"):
        stats._calc_stat(np.asarray([1.0]), np.asarray([2.0]), "unsupported")


@pytest.mark.parametrize(
    ("runs", "conditions", "evaluations", "message"),
    [
        ([{"condition": "a", "score": 1.0}], None, None, "missing 'run_id'"),
        ([{"run_id": "r1", "condition": "", "score": 1.0}], None, None, "direct condition"),
        ([{"run_id": "r1", "condition_id": "c1", "score": 1.0}], None, None, "no conditions table"),
        (
            [{"run_id": "r1", "score": 1.0}],
            [{"condition_id": "c1", "condition": "a"}],
            None,
            "conditions join",
        ),
        (
            [{"run_id": "r1", "condition_id": "missing", "score": 1.0}],
            [{"condition_id": "c1", "condition": "a"}],
            None,
            "unknown condition_id",
        ),
        (
            [{"run_id": "r1", "condition_id": "c1", "score": 1.0}],
            [{"condition_id": "c1"}],
            None,
            "is missing 'condition'",
        ),
        ([{"run_id": "r1", "condition": "a", "score": ""}], None, None, "missing metric column"),
        ([{"run_id": "r1", "condition": "a"}], None, None, "no evaluations table"),
        (
            [{"run_id": "r1", "condition": "a"}],
            None,
            [{"run_id": "r1", "metric_name": "other", "metric_value": 1.0}],
            "No evaluation metric",
        ),
        (
            [{"run_id": "r1", "condition": "a"}],
            None,
            [
                {"run_id": "r1", "metric_name": "score", "metric_value": 1.0},
                {"run_id": "r1", "metric_name": "score", "metric_value": 2.0},
            ],
            "Multiple evaluation rows",
        ),
        (
            [{"run_id": "r1", "condition": "a"}],
            None,
            [{"run_id": "r1", "metric_name": "score", "metric_value": ""}],
            "is missing 'metric_value'",
        ),
    ],
)
def test_condition_metric_table_reports_join_failures(
    runs: list[dict[str, object]],
    conditions: list[dict[str, object]] | None,
    evaluations: list[dict[str, object]] | None,
    message: str,
) -> None:
    """Malformed experiment artifacts should identify the failed join contract."""
    with pytest.raises(ValueError, match=message):
        stats.build_condition_metric_table(
            runs,
            metric="score",
            conditions=conditions,
            evaluations=evaluations,
        )


def test_decode_results_support_serialization_and_comparison() -> None:
    """Decoded HMM state paths should participate in the shared result algebra."""
    left = sequence_models.DecodeResult(
        algorithm="viterbi",
        log_probability=-1.0,
        states=np.asarray([0, 1, 1]),
        lengths=[3],
        backend="fake",
    )
    right = sequence_models.DecodeResult(
        algorithm="map",
        log_probability=-2.0,
        states=np.asarray([0, 0, 1]),
        lengths=None,
        backend="fake",
    )

    assert left.to_dict()["states"] == [0, 1, 1]
    assert (left - right).details["algorithms"] == ["viterbi", "map"]


def test_sequence_comparison_helpers_validate_degenerate_inputs() -> None:
    """Internal comparison helpers should define behavior for degenerate tables."""
    with pytest.raises(ValueError, match="2D contingency"):
        sequence_models._chi_square_statistic(np.ones(3))
    assert sequence_models._chi_square_statistic(np.ones((1, 2))) == (0.0, 0)
    assert sequence_models._chi_square_statistic(np.zeros((2, 2))) == (0.0, 0)
    assert sequence_models._chi_square_p_value(1.0, 0) is None


@pytest.mark.skipif(importlib.util.find_spec("hmmlearn") is None, reason="hmmlearn unavailable")
def test_hmm_table_adapters_cover_discrete_and_embedded_text() -> None:
    """Unified tables should feed both discrete and text Gaussian HMM adapters."""
    event_rows = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "session_id": "s1",
            "actor_id": "a",
            "event_type": "frame",
        },
        {
            "timestamp": "2026-01-01T00:00:01Z",
            "session_id": "s1",
            "actor_id": "b",
            "event_type": "idea",
        },
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "session_id": "s2",
            "actor_id": "a",
            "event_type": "frame",
        },
        {
            "timestamp": "2026-01-01T00:00:01Z",
            "session_id": "s2",
            "actor_id": "b",
            "event_type": "evaluate",
        },
    ]
    discrete = sequence_models.fit_discrete_hmm_from_table(
        event_rows,
        n_states=2,
        n_iter=10,
        seed=3,
        include_actor_in_token=True,
    )
    assert discrete.config["source"] == "table"

    text_rows = [
        {"timestamp": "2026-01-01T00:00:00Z", "session_id": "s1", "text": "frame problem"},
        {"timestamp": "2026-01-01T00:00:01Z", "session_id": "s1", "text": "generate idea"},
        {"timestamp": "2026-01-01T00:00:00Z", "session_id": "s2", "text": "inspect concept"},
        {"timestamp": "2026-01-01T00:00:01Z", "session_id": "s2", "text": "evaluate concept"},
    ]

    def embedder(texts: list[str]) -> np.ndarray:
        return np.asarray([[len(text), text.count("e")] for text in texts], dtype=float)

    gaussian = sequence_models.fit_text_gaussian_hmm_from_table(
        text_rows,
        n_states=2,
        embedder=embedder,
        n_iter=10,
        seed=3,
    )
    assert gaussian.config["source"] == "table"
    assert gaussian.config["n_sessions"] == 2


def test_cli_data_helpers_reject_ambiguous_and_malformed_inputs(tmp_path: Path) -> None:
    """CLI helper errors should be specific before expensive analysis begins."""
    assert cli._serialize_for_json(datetime(2026, 1, 1, tzinfo=UTC)).startswith("2026-01-01")
    assert cli._serialize_for_json((np.asarray([1, 2]),)) == [[1, 2]]

    empty_csv = tmp_path / "empty.csv"
    cli._write_csv(str(empty_csv), [])
    assert empty_csv.read_text(encoding="utf-8") == ""

    assert cli._extract_record_ids([{}, {}], record_id_column="id") == ["0", "1"]
    with pytest.raises(ValueError, match="must be unique"):
        cli._extract_record_ids([{"id": "x"}, {"id": "x"}], record_id_column="id")
    with pytest.raises(ValueError, match="at least one column"):
        cli._feature_matrix_from_rows([], feature_columns=[], record_id_column="id")
    with pytest.raises(ValueError, match="missing 'x'"):
        cli._feature_matrix_from_rows([{"id": "a"}], feature_columns=["x"], record_id_column="id")
    with pytest.raises(ValueError, match="non-numeric 'x'"):
        cli._feature_matrix_from_rows(
            [{"id": "a", "x": "bad"}], feature_columns=["x"], record_id_column="id"
        )

    embedding = EmbeddingMapResult(np.zeros((2, 2)), ["a", "missing"], "pca")
    with pytest.raises(ValueError, match="missing record IDs"):
        cli._aligned_rows_for_map(embedding, [{"id": "a"}], record_id_column="id")
    with pytest.raises(ValueError, match="provided together"):
        cli._trajectory_summary_for_map(
            embedding,
            [{"id": "a"}],
            record_id_column="id",
            group_column="group",
            order_column=None,
        )
    default_trajectory = cli._trajectory_summary_for_map(
        EmbeddingMapResult(np.asarray([[0.0, 0.0], [1.0, 1.0]]), ["a", "b"], "pca"),
        [{"id": "a"}, {"id": "b"}],
        record_id_column="id",
        group_column=None,
        order_column=None,
    )
    assert "divergence_convergence" in default_trajectory

    with pytest.raises(ValueError, match="Mapper spec"):
        cli._load_mapper("invalid")
    with pytest.raises(ValueError, match="did not resolve"):
        cli._load_mapper("math:missing")
    assert cli._load_mapper("math.sqrt") is not None

    with pytest.raises(ValueError, match="Invalid options"):
        cli._parse_json_object("{", label="options")
    with pytest.raises(ValueError, match="must decode to a JSON object"):
        cli._parse_json_object("[]", label="options")

    with pytest.raises(ValueError, match="file not found"):
        cli._load_json_object_file(str(tmp_path / "missing.json"), label="options")
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid options file"):
        cli._load_json_object_file(str(invalid_file), label="options")
    list_file = tmp_path / "list.json"
    list_file.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        cli._load_json_object_file(str(list_file), label="options")
    object_file = tmp_path / "object.json"
    object_file.write_text('{"seed": 3}', encoding="utf-8")
    assert cli._load_json_object_file(str(object_file), label="options") == {"seed": 3}
    assert cli._resolve_json_object_source(
        inline_json=None,
        json_file=str(object_file),
        label="options",
    ) == {"seed": 3}
    assert (
        cli._resolve_json_object_source(
            inline_json=None,
            json_file=None,
            label="options",
        )
        is None
    )

    with pytest.raises(ValueError, match="either inline"):
        cli._resolve_json_object_source(
            inline_json="{}",
            json_file=str(list_file),
            label="options",
        )
    with pytest.raises(ValueError, match="is required"):
        cli._resolve_json_object_source(
            inline_json=None,
            json_file=None,
            label="options",
            required=True,
        )


def test_cli_rethrows_unknown_table_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only known input-format errors should be rewritten by the CLI."""

    def fail(_path: str) -> list[dict[str, object]]:
        raise ValueError("unexpected validation failure")

    monkeypatch.setattr(cli, "coerce_unified_table", fail)
    with pytest.raises(ValueError, match="unexpected validation failure"):
        cli._load_table("input.csv")


def test_cli_dataframe_loader_rewrites_only_known_format_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset loading should preserve unknown validation failures."""
    monkeypatch.setattr(cli, "_load_pandas", lambda: (SimpleNamespace(), object()))
    monkeypatch.setattr(
        cli,
        "_coerce_dataframe_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError(cli._DATASET_INPUT_ERROR)),
    )
    with pytest.raises(ValueError, match="Unsupported dataset input format"):
        cli._load_dataframe("input.bad")

    monkeypatch.setattr(
        cli,
        "_coerce_dataframe_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("unknown dataframe error")),
    )
    with pytest.raises(ValueError, match="unknown dataframe error"):
        cli._load_dataframe("input.csv")
