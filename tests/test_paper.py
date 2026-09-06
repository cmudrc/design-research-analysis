from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from design_research_analysis import (
    ANALYSIS_RESULT_VERSION,
    PAPER_CONTRIBUTION_VERSION,
    AnalysisCheck,
    AnalysisExclusion,
    AnalysisResultRecord,
    AnalysisStatus,
    build_analysis_result,
    collect_analysis_paper_contributions,
    load_analysis_result,
    write_analysis_result,
)
from design_research_analysis._comparison import ComparisonResult
from design_research_analysis.embedding_maps import EmbeddingMapResult
from design_research_analysis.language import LanguageConvergenceResult
from design_research_analysis.reliability import InterraterReliabilityResult
from design_research_analysis.sequence import fit_markov_chain
from design_research_analysis.stats import (
    ConditionComparisonReport,
    ConditionPairComparison,
    GroupComparisonResult,
    MixedEffectsResult,
    RegressionResult,
)


def _regression() -> RegressionResult:
    return RegressionResult(
        coefficients={"model_size": 0.42, "temperature": -0.1},
        intercept=1.2,
        r2=0.31,
        mse=0.08,
        n_samples=35,
        n_features=2,
        config={"outcome": "quality", "add_intercept": True},
    )


def _record(result: object | None = None, **kwargs: object) -> AnalysisResultRecord:
    defaults: dict[str, object] = {
        "analysis_id": "h1-regression",
        "hypothesis_ids": ("H1",),
        "candidate_run_ids": ("run-1", "run-2"),
        "included_run_ids": ("run-1",),
        "exclusions": ({"run_id": "run-2", "reason": "Missing evaluator output."},),
        "evidence_refs": ("artifacts/runs/run-1/run.json",),
        "source_api": "fit_regression_from_artifacts",
    }
    defaults.update(kwargs)
    return build_analysis_result(_regression() if result is None else result, **defaults)


def _gap_ids(packet: dict[str, object]) -> set[str]:
    gaps = packet["reporting_gaps"]
    assert isinstance(gaps, list)
    return {str(item["gap_id"]) for item in gaps}


def _contribution_ids(packet: dict[str, object]) -> set[str]:
    contributions = packet["contributions"]
    assert isinstance(contributions, list)
    return {str(item["contribution_id"]) for item in contributions}


def test_versions_and_regression_record_are_stable() -> None:
    record = _record(
        analysis_plan_ids=("plan-1",),
        assumptions=(AnalysisCheck("linearity", "passed", "Residual plot was inspected."),),
        diagnostics=(
            {"check_id": "influence", "status": "not-applicable", "detail": "No leverage."},
        ),
        tables=("artifacts/analysis/tables/h1.tex",),
        figures=("artifacts/analysis/figures/h1.pdf",),
        decision_threshold=0.05,
    )

    payload = record.to_dict()

    assert ANALYSIS_RESULT_VERSION == "0.1.0"
    assert PAPER_CONTRIBUTION_VERSION == "0.1.0"
    assert payload["analysis_result_version"] == "0.1.0"
    assert payload["status"] == "complete"
    assert payload["method_id"] == "analysis.stats.ols"
    assert payload["parameters"] == {"add_intercept": True, "outcome": "quality"}
    assert payload["statistics"]["coefficients"]["model_size"] == 0.42
    assert payload["exclusions"] == [{"run_id": "run-2", "reason": "Missing evaluator output."}]
    assert AnalysisResultRecord.from_mapping(payload) == record


def test_write_and_load_analysis_result_are_explicit_and_atomic(tmp_path: Path) -> None:
    record = _record()

    path = write_analysis_result(record, output_dir=tmp_path)

    assert path == tmp_path / "artifacts/analysis/results/h1-regression.json"
    assert path.exists()
    assert not list(path.parent.glob("*.tmp"))
    assert load_analysis_result(path) == record
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_analysis_result(record, output_dir=tmp_path)

    replacement = _record(parameters={"author_note": "verified"})
    assert write_analysis_result(replacement, output_dir=tmp_path, overwrite=True) == path
    assert load_analysis_result(path).parameters["author_note"] == "verified"


def test_load_analysis_result_rejects_non_object_and_unknown_version(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        load_analysis_result(path)

    path.write_text('{"analysis_result_version": "9.9.9"}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_analysis_result(path)


def test_regression_packet_is_analyzed_and_retains_hypothesis_links() -> None:
    packet = collect_analysis_paper_contributions(_record())

    assert packet["schema_version"] == "0.1.0"
    assert packet["source"]["package"] == "design-research-analysis"
    assert packet["source"]["component_id"] == "analysis.stats.ols"
    assert _contribution_ids(packet) == {
        "analysis:h1-regression:methods",
        "analysis:h1-regression:results:summary",
    }
    methods, results = packet["contributions"]
    assert methods["evidence_basis"] == "analyzed"
    assert methods["metadata"]["hypothesis_ids"] == ["H1"]
    assert "ordinary least-squares" in methods["text"]
    assert "R²=0.31" in results["text"]
    assert "model_size=0.42" in results["text"]
    assert "analysis:h1-regression:regression-uncertainty" in _gap_ids(packet)


def test_group_comparison_reports_threshold_only_when_prespecified() -> None:
    result = GroupComparisonResult(
        method="ttest",
        statistic=-3.2,
        p_value=0.003,
        effect_size=-0.8,
        group_means={"A": 1.0, "B": 2.0},
        group_sizes={"A": 20, "B": 20},
        config={"alternative": "two-sided"},
    )
    record = build_analysis_result(
        result,
        analysis_id="group-test",
        evidence_refs=("evaluations.csv",),
        decision_threshold=0.05,
    )
    packet = collect_analysis_paper_contributions(record)

    assert record.method_id == "analysis.stats.group-comparison.ttest"
    assert "met the prespecified alpha=0.05 threshold" in packet["contributions"][1]["text"]

    no_threshold = build_analysis_result(
        result,
        analysis_id="group-test-no-alpha",
        evidence_refs=("evaluations.csv",),
    )
    no_threshold_text = collect_analysis_paper_contributions(no_threshold)["contributions"][1][
        "text"
    ]
    assert "prespecified" not in no_threshold_text


def test_condition_comparison_emits_one_factual_result_per_pair() -> None:
    comparisons = (
        ConditionPairComparison(
            metric="quality",
            left_condition="A",
            right_condition="B",
            mean_left=3.0,
            mean_right=2.0,
            n_left=10,
            n_right=10,
            mean_difference=1.0,
            effect_size=0.5,
            p_value=0.02,
            alternative="two-sided",
            test_method="exact",
            permutations_evaluated=184756,
            total_permutations=184756,
            higher_condition="A",
            significant=True,
        ),
        ConditionPairComparison(
            metric="quality",
            left_condition="A",
            right_condition="C",
            mean_left=3.0,
            mean_right=3.1,
            n_left=10,
            n_right=10,
            mean_difference=-0.1,
            effect_size=-0.04,
            p_value=0.7,
            alternative="two-sided",
            test_method="sampled",
            permutations_evaluated=20000,
            total_permutations=184756,
            higher_condition="C",
            significant=False,
        ),
    )
    result = ConditionComparisonReport(
        metric="quality",
        condition_column="condition_id",
        metric_column="value",
        alternative="two-sided",
        alpha=0.05,
        comparisons=comparisons,
        config={"seed": 7},
    )
    record = build_analysis_result(
        result,
        analysis_id="condition-pairs",
        included_run_ids=("run-1", "run-2"),
        evidence_refs=("evaluations.csv",),
    )
    packet = collect_analysis_paper_contributions(record)

    assert record.method_id == "analysis.stats.condition-comparison.permutation"
    assert len(packet["contributions"]) == 3
    assert "A vs B" in packet["contributions"][1]["text"]
    assert packet["contributions"][1]["kind"] == "bullet"
    assert "sampled permutation test" in packet["contributions"][2]["text"]


@pytest.mark.parametrize(
    ("method", "expected_key"),
    [
        ("cohen_kappa", "cohen1960"),
        ("fleiss_kappa", "fleiss1971"),
        ("krippendorff_alpha", "krippendorff2018"),
    ],
)
def test_reliability_contributions_use_curated_references(
    method: str,
    expected_key: str,
) -> None:
    result = InterraterReliabilityResult(
        method=method,
        coefficient=0.72,
        n_items=24,
        n_items_used=22,
        n_raters=3,
        n_observations=65,
        categories=("a", "b"),
        missing_ratings=7,
        confidence_interval=(0.61, 0.81),
        bootstrap_samples=1000,
        config={"missing_policy": "complete_case"},
    )
    record = build_analysis_result(
        result,
        analysis_id=f"reliability-{method}",
        evidence_refs=("coded-events.csv",),
    )
    packet = collect_analysis_paper_contributions(record)

    assert expected_key in packet["contributions"][0]["citation_keys"]
    assert packet["references"][0]["key"] == expected_key
    assert packet["references"][0]["raw_text"].startswith("@")
    assert "0.72" in packet["contributions"][1]["text"]
    assert not any(gap.endswith("reliability-interval") for gap in _gap_ids(packet))


def test_reliability_without_interval_produces_visible_gap() -> None:
    result = InterraterReliabilityResult(
        method="cohen_kappa",
        coefficient=0.6,
        n_items=10,
        n_items_used=10,
        n_raters=2,
        n_observations=20,
        categories=("a", "b"),
        missing_ratings=0,
    )
    packet = collect_analysis_paper_contributions(
        build_analysis_result(
            result,
            analysis_id="reliability-no-ci",
            evidence_refs=("codings.csv",),
        )
    )

    assert "analysis:reliability-no-ci:reliability-interval" in _gap_ids(packet)


def test_markov_fit_and_comparison_are_supported() -> None:
    left = fit_markov_chain([["A", "B", "A"], ["A", "A", "B"]], smoothing=1.0)
    right = fit_markov_chain([["A", "B", "B"], ["B", "A", "B"]], smoothing=1.0)

    fit_record = build_analysis_result(
        left,
        analysis_id="markov-fit",
        evidence_refs=("events.csv",),
    )
    comparison_record = build_analysis_result(
        left - right,
        analysis_id="markov-comparison",
        evidence_refs=("events.csv",),
        decision_threshold=0.05,
    )

    assert fit_record.method_id == "analysis.sequence.markov-chain"
    assert (
        "order-1 Markov chain"
        in collect_analysis_paper_contributions(fit_record)["contributions"][1]["text"]
    )
    assert comparison_record.method_id == "analysis.sequence.markov-comparison"
    comparison_text = collect_analysis_paper_contributions(comparison_record)["contributions"][1][
        "text"
    ]
    assert "transition_profile comparison estimate" in comparison_text
    assert "prespecified alpha=0.05" in comparison_text


def test_generic_comparison_and_unknown_result_renderer_are_honest() -> None:
    comparison = ComparisonResult(
        operation="difference",
        left_type="EmbeddingResult",
        right_type="EmbeddingResult",
        metric="embedding_profile",
        estimate=0.25,
        statistic=0.25,
        p_value=None,
        effect_size=0.1,
    )
    generic = build_analysis_result(
        comparison,
        analysis_id="embedding-comparison",
        evidence_refs=("embeddings.json",),
    )
    assert generic.method_id == "analysis.result-comparison"
    assert (
        "comparison estimate was 0.25"
        in collect_analysis_paper_contributions(generic)["contributions"][1]["text"]
    )

    unknown = build_analysis_result(
        {"estimate": 3.0},
        analysis_id="custom-analysis",
        method_id="analysis.custom.method",
        evidence_refs=("custom-result.json",),
    )
    packet = collect_analysis_paper_contributions(unknown)
    assert "analysis:custom-analysis:unsupported-result-renderer" in _gap_ids(packet)
    assert "no more specific curated method description" in packet["contributions"][0]["text"]


def test_dataset_profile_and_profile_warnings_are_rendered() -> None:
    profile = {
        "n_rows": 40,
        "n_columns": 6,
        "columns": {"quality": {"mean": 3.1, "missing_count": 2}},
        "warnings": ["quality contains two missing values"],
    }
    record = build_analysis_result(
        profile,
        analysis_id="dataset-summary",
        evidence_refs=("supporting-data/analysis.csv",),
    )
    packet = collect_analysis_paper_contributions(record)

    assert record.result_type == "DatasetProfile"
    assert record.method_id == "analysis.dataset.profile"
    assert "40 rows and 6 columns" in packet["contributions"][1]["text"]
    assert "1 warning(s)" in packet["contributions"][1]["text"]


def test_embedding_map_attaches_figure_and_curated_umap_reference() -> None:
    result = EmbeddingMapResult(
        coordinates=np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]),
        record_ids=["a", "b", "c"],
        method="umap",
        config={"n_neighbors": 2, "random_state": 7},
    )
    record = build_analysis_result(
        result,
        analysis_id="idea-map",
        figures=("artifacts/analysis/figures/idea-map.pdf",),
        evidence_refs=("embeddings.json",),
    )
    packet = collect_analysis_paper_contributions(record)

    assert record.method_id == "analysis.embedding-map.umap"
    assert packet["references"][0]["key"] == "mcinnes2018umap"
    assert packet["contributions"][0]["citation_keys"] == ["mcinnes2018umap"]
    assert packet["contributions"][2]["kind"] == "figure"
    assert packet["contributions"][2]["metadata"]["path"].endswith("idea-map.pdf")
    assert not any(gap.endswith("embedding-map-figure") for gap in _gap_ids(packet))


def test_embedding_map_without_figure_and_pca_without_reference() -> None:
    result = EmbeddingMapResult(
        coordinates=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        record_ids=["a", "b"],
        method="pca",
    )
    packet = collect_analysis_paper_contributions(
        build_analysis_result(
            result,
            analysis_id="pca-map",
            evidence_refs=("features.csv",),
        )
    )

    assert packet["references"] == []
    assert "analysis:pca-map:embedding-map-figure" in _gap_ids(packet)


def test_language_convergence_is_supported() -> None:
    result = LanguageConvergenceResult(
        groups=["A", "B"],
        distance_trajectories={"A": [1.0, 0.8], "B": [1.0, 1.1]},
        slope_by_group={"A": -0.2, "B": 0.1},
        direction_by_group={"A": "converging", "B": "diverging"},
        window_size=2,
        n_observations=8,
        config={"embedding_provider": "callable"},
    )
    packet = collect_analysis_paper_contributions(
        build_analysis_result(
            result,
            analysis_id="language-convergence",
            evidence_refs=("events.csv",),
        )
    )

    assert packet["source"]["component_id"] == "analysis.language.convergence"
    assert "2 group(s) across 8 observations" in packet["contributions"][1]["text"]


def test_mixed_effects_success_and_backend_failure_are_distinct() -> None:
    success = MixedEffectsResult(
        success=True,
        backend="statsmodels",
        formula="quality ~ condition",
        group_column="participant",
        params={"Intercept": 1.2, "condition": 0.4},
        aic=12.0,
        bic=13.0,
        log_likelihood=-4.0,
        config={"reml": True},
    )
    success_packet = collect_analysis_paper_contributions(
        build_analysis_result(
            success,
            analysis_id="mixed-fit",
            evidence_refs=("runs.csv",),
        )
    )
    assert "fit successfully" in success_packet["contributions"][1]["text"]

    unsuccessful = MixedEffectsResult(
        success=False,
        backend="statsmodels",
        formula="quality ~ condition",
        group_column="participant",
        params={},
        aic=None,
        bic=None,
        log_likelihood=None,
        message="did not converge",
    )
    failed_packet = collect_analysis_paper_contributions(
        build_analysis_result(
            unsuccessful,
            analysis_id="mixed-backend-failure",
            evidence_refs=("runs.csv",),
        )
    )
    assert "analysis:mixed-backend-failure:mixed-effects-fit-failed" in _gap_ids(failed_packet)
    assert "analysis:mixed-backend-failure:unsupported-result-renderer" in _gap_ids(failed_packet)


def test_failed_and_planned_analyses_emit_todos_not_results() -> None:
    failed = build_analysis_result(
        None,
        analysis_id="failed-analysis",
        method_id="analysis.stats.ols",
        status="failed",
        error={"type": "RuntimeError", "message": "singular matrix"},
    )
    failed_packet = collect_analysis_paper_contributions(failed)
    assert _contribution_ids(failed_packet) == {"analysis:failed-analysis:methods"}
    assert failed_packet["contributions"][0]["evidence_basis"] == "configured"
    assert "singular matrix" in failed_packet["reporting_gaps"][0]["message"]

    planned = build_analysis_result(
        None,
        analysis_id="planned-analysis",
        method_id="analysis.sequence.markov-chain",
        status=AnalysisStatus.PLANNED,
        analysis_plan_ids=("plan-2",),
    )
    planned_packet = collect_analysis_paper_contributions(planned.to_dict())
    assert "analysis:planned-analysis:analysis-not-executed" in _gap_ids(planned_packet)
    assert planned.statistics == {}


def test_failed_assumption_diagnostic_and_unaccounted_runs_become_gaps() -> None:
    record = _record(
        candidate_run_ids=("run-1", "run-2", "run-3"),
        included_run_ids=("run-1",),
        exclusions=({"run_id": "run-2", "reason": "Missing score."},),
        assumptions=(
            {"check_id": "linearity", "status": "failed", "detail": "Residual curvature."},
        ),
        diagnostics=(
            {"check_id": "influence", "status": "failed", "detail": "One high-leverage run."},
        ),
    )
    gaps = _gap_ids(collect_analysis_paper_contributions(record))

    assert "analysis:h1-regression:assumption-linearity" in gaps
    assert "analysis:h1-regression:diagnostic-influence" in gaps
    assert "analysis:h1-regression:unaccounted-candidate-runs" in gaps


def test_table_and_figure_paths_produce_typed_contributions() -> None:
    record = _record(
        tables=("artifacts/analysis/tables/h1.tex", "artifacts/analysis/tables/h1.csv"),
        figures=("artifacts/analysis/figures/h1.pdf",),
    )
    packet = collect_analysis_paper_contributions(record)
    kinds = [item["kind"] for item in packet["contributions"]]

    assert kinds == ["paragraph", "paragraph", "table", "table", "figure"]
    assert packet["contributions"][-1]["evidence_refs"] == ["artifacts/runs/run-1/run.json"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"analysis_id": "../escape"}, "analysis_id"),
        ({"evidence_refs": ()}, "evidence_ref"),
        (
            {
                "candidate_run_ids": ("run-1",),
                "included_run_ids": ("run-2",),
                "exclusions": (),
            },
            "candidate",
        ),
        (
            {
                "candidate_run_ids": ("run-1",),
                "included_run_ids": ("run-1",),
                "exclusions": ({"run_id": "run-1", "reason": "bad"},),
            },
            "both included and excluded",
        ),
        ({"tables": ("../escape.tex",)}, "safe relative"),
        ({"figures": ("/absolute.pdf",)}, "safe relative"),
        ({"decision_threshold": 1.0}, "between 0 and 1"),
        ({"candidate_run_ids": ("run-1", "run-1")}, "duplicates"),
    ],
)
def test_result_record_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _record(**kwargs)


def test_exclusion_and_check_validation() -> None:
    with pytest.raises(ValueError, match="run_id"):
        AnalysisExclusion("", "reason")
    with pytest.raises(ValueError, match="reason"):
        AnalysisExclusion("run-1", "")
    with pytest.raises(ValueError, match="check_id"):
        AnalysisCheck("", "passed", "fine")
    with pytest.raises(ValueError, match="status"):
        AnalysisCheck("normality", "missing", "unknown")
    with pytest.raises(ValueError, match="detail"):
        AnalysisCheck("normality", "passed", "")


def test_lifecycle_validation_rejects_inconsistent_payloads() -> None:
    base = _record().to_dict()
    base["status"] = "complete"
    base["error"] = {"message": "impossible"}
    with pytest.raises(ValueError, match="complete analysis cannot"):
        AnalysisResultRecord.from_mapping(base)

    base = _record().to_dict()
    base["status"] = "failed"
    base["error"] = None
    with pytest.raises(ValueError, match="failed analysis requires"):
        AnalysisResultRecord.from_mapping(base)

    base = _record().to_dict()
    base["status"] = "planned"
    with pytest.raises(ValueError, match="planned analysis cannot"):
        AnalysisResultRecord.from_mapping(base)

    base = _record().to_dict()
    base["status"] = "unknown"
    with pytest.raises(ValueError, match="planned, complete, failed"):
        AnalysisResultRecord.from_mapping(base)

    record = _record()
    with pytest.raises(ValueError, match="Unsupported analysis result version"):
        replace(record, analysis_result_version="9.9.9")
    with pytest.raises(ValueError, match="method_id"):
        replace(record, method_id="")
    with pytest.raises(ValueError, match="result_type"):
        replace(record, result_type="")
    with pytest.raises(ValueError, match="AnalysisStatus"):
        replace(record, status="complete")  # type: ignore[arg-type]


def test_build_analysis_result_requires_known_method_and_valid_result() -> None:
    with pytest.raises(ValueError, match="complete analysis requires"):
        build_analysis_result(
            None,
            analysis_id="missing-result",
            method_id="analysis.custom",
            evidence_refs=("data.csv",),
        )
    with pytest.raises(ValueError, match="method_id is required"):
        build_analysis_result(
            {"estimate": 1.0},
            analysis_id="custom-result",
            evidence_refs=("data.csv",),
        )
    with pytest.raises(TypeError, match="supported result objects"):
        build_analysis_result(
            object(),
            analysis_id="bad-result",
            method_id="analysis.custom",
            evidence_refs=("data.csv",),
        )

    class _BadResult:
        def to_dict(self) -> list[object]:
            return []

    with pytest.raises(TypeError, match=r"to_dict\(\) must return a mapping"):
        build_analysis_result(
            _BadResult(),
            analysis_id="bad-to-dict",
            method_id="analysis.custom",
            evidence_refs=("data.csv",),
        )


def test_mapping_loader_validates_nested_shapes() -> None:
    payload = _record().to_dict()
    payload["statistics"] = []
    with pytest.raises(ValueError, match="statistics"):
        AnalysisResultRecord.from_mapping(payload)

    payload = _record().to_dict()
    payload["exclusions"] = ["bad"]
    with pytest.raises(ValueError, match="JSON objects"):
        AnalysisResultRecord.from_mapping(payload)

    payload = _record().to_dict()
    payload["tables"] = "not-a-sequence"
    with pytest.raises(ValueError, match="sequence of strings"):
        AnalysisResultRecord.from_mapping(payload)

    payload = _record().to_dict()
    payload["hypothesis_ids"] = [""]
    with pytest.raises(ValueError, match="blank"):
        AnalysisResultRecord.from_mapping(payload)


def test_packet_accepts_mapping_record_and_rejects_other_inputs() -> None:
    record = _record()
    assert collect_analysis_paper_contributions(record.to_dict())["schema_version"] == "0.1.0"
    with pytest.raises(TypeError, match="Analysis result"):
        collect_analysis_paper_contributions([])  # type: ignore[arg-type]


def test_json_round_trip_retains_numpy_scalars_and_arrays(tmp_path: Path) -> None:
    record = build_analysis_result(
        {
            "estimate": np.float64(0.25),
            "counts": np.asarray([1, 2, 3]),
        },
        analysis_id="numpy-payload",
        method_id="analysis.custom.numpy",
        evidence_refs=("values.npy",),
    )
    path = write_analysis_result(record, output_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["statistics"] == {"counts": [1, 2, 3], "estimate": 0.25}
