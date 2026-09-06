"""Evidence-linked analysis records and deterministic paper contributions."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ._comparison import ComparisonResult
from ._version import __version__
from .embedding_maps import EmbeddingMapResult
from .language import LanguageConvergenceResult
from .reliability import InterraterReliabilityResult
from .sequence import MarkovChainResult
from .stats import (
    ConditionComparisonReport,
    GroupComparisonResult,
    MixedEffectsResult,
    RegressionResult,
)

ANALYSIS_RESULT_VERSION = "0.1.0"
PAPER_CONTRIBUTION_VERSION = "0.1.0"

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_TERMINAL_CHECK_STATUSES = {"passed", "failed", "not-applicable"}


class AnalysisStatus(StrEnum):
    """Lifecycle state for one planned or executed analysis."""

    PLANNED = "planned"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AnalysisExclusion:
    """One candidate run excluded from an analysis, with a required reason."""

    run_id: str
    reason: str

    def __post_init__(self) -> None:
        """Require an identified run and an honest exclusion reason."""
        if not self.run_id.strip():
            raise ValueError("Analysis exclusion run_id must be non-empty.")
        if not self.reason.strip():
            raise ValueError("Analysis exclusion reason must be non-empty.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AnalysisExclusion:
        """Build an exclusion from its JSON-compatible representation."""
        return cls(
            run_id=_required_text(payload, "run_id", context="analysis exclusion"),
            reason=_required_text(payload, "reason", context="analysis exclusion"),
        )

    def to_dict(self) -> dict[str, str]:
        """Return the stable JSON representation."""
        return {"run_id": self.run_id, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class AnalysisCheck:
    """Recorded assumption or diagnostic check for an executed analysis."""

    check_id: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        """Require a stable check identity and a recognized status."""
        if not self.check_id.strip():
            raise ValueError("Analysis check check_id must be non-empty.")
        if self.status not in _TERMINAL_CHECK_STATUSES:
            raise ValueError(
                "Analysis check status must be one of: passed, failed, not-applicable."
            )
        if not self.detail.strip():
            raise ValueError("Analysis check detail must be non-empty.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AnalysisCheck:
        """Build a check from its JSON-compatible representation."""
        return cls(
            check_id=_required_text(payload, "check_id", context="analysis check"),
            status=_required_text(payload, "status", context="analysis check"),
            detail=_required_text(payload, "detail", context="analysis check"),
        )

    def to_dict(self) -> dict[str, str]:
        """Return the stable JSON representation."""
        return {
            "check_id": self.check_id,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class AnalysisResultRecord:
    """Portable record of one planned, complete, or failed analysis."""

    analysis_id: str
    method_id: str
    status: AnalysisStatus
    result_type: str
    statistics: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    analysis_plan_ids: tuple[str, ...] = ()
    hypothesis_ids: tuple[str, ...] = ()
    candidate_run_ids: tuple[str, ...] = ()
    included_run_ids: tuple[str, ...] = ()
    exclusions: tuple[AnalysisExclusion, ...] = ()
    assumptions: tuple[AnalysisCheck, ...] = ()
    diagnostics: tuple[AnalysisCheck, ...] = ()
    tables: tuple[str, ...] = ()
    figures: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    source_api: str = ""
    decision_threshold: float | None = None
    error: Mapping[str, Any] | None = None
    analysis_result_version: str = ANALYSIS_RESULT_VERSION

    def __post_init__(self) -> None:
        """Validate identity, accounting, artifact safety, and lifecycle consistency."""
        if self.analysis_result_version != ANALYSIS_RESULT_VERSION:
            raise ValueError("Unsupported analysis result version.")
        if not _SAFE_ID.fullmatch(self.analysis_id):
            raise ValueError(
                "analysis_id must use only letters, numbers, dots, underscores, or hyphens."
            )
        if not self.method_id.strip():
            raise ValueError("method_id must be non-empty.")
        if not self.result_type.strip():
            raise ValueError("result_type must be non-empty.")
        if not isinstance(self.status, AnalysisStatus):
            raise ValueError("status must be an AnalysisStatus value.")
        _require_unique(self.analysis_plan_ids, field_name="analysis_plan_ids")
        _require_unique(self.hypothesis_ids, field_name="hypothesis_ids")
        _require_unique(self.candidate_run_ids, field_name="candidate_run_ids")
        _require_unique(self.included_run_ids, field_name="included_run_ids")
        exclusion_ids = tuple(item.run_id for item in self.exclusions)
        _require_unique(exclusion_ids, field_name="exclusion run IDs")
        if set(self.included_run_ids) & set(exclusion_ids):
            raise ValueError("A run cannot be both included and excluded from an analysis.")
        if self.candidate_run_ids:
            accounted = set(self.included_run_ids) | set(exclusion_ids)
            unknown = accounted - set(self.candidate_run_ids)
            if unknown:
                raise ValueError(
                    "Included and excluded runs must belong to candidate_run_ids: "
                    + ", ".join(sorted(unknown))
                )
        for artifact_path in (*self.tables, *self.figures):
            _validate_relative_artifact_path(artifact_path)
        if self.decision_threshold is not None and not 0.0 < self.decision_threshold < 1.0:
            raise ValueError("decision_threshold must be between 0 and 1.")
        if self.status == AnalysisStatus.COMPLETE:
            if self.error is not None:
                raise ValueError("A complete analysis cannot include an error.")
            if not self.evidence_refs:
                raise ValueError("A complete analysis requires at least one evidence_ref.")
        elif self.status == AnalysisStatus.FAILED:
            if not self.error:
                raise ValueError("A failed analysis requires a structured error.")
        elif self.statistics:
            raise ValueError("A planned analysis cannot include result statistics.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AnalysisResultRecord:
        """Validate and load a JSON-compatible analysis result record."""
        version = str(payload.get("analysis_result_version", ""))
        if version != ANALYSIS_RESULT_VERSION:
            raise ValueError(
                f"analysis_result_version {version!r} does not match {ANALYSIS_RESULT_VERSION!r}."
            )
        return cls(
            analysis_id=_required_text(payload, "analysis_id", context="analysis result"),
            method_id=_required_text(payload, "method_id", context="analysis result"),
            status=_coerce_status(payload.get("status")),
            result_type=_required_text(payload, "result_type", context="analysis result"),
            statistics=_mapping(payload.get("statistics", {}), field_name="statistics"),
            parameters=_mapping(payload.get("parameters", {}), field_name="parameters"),
            analysis_plan_ids=_string_tuple(
                payload.get("analysis_plan_ids", ()), field_name="analysis_plan_ids"
            ),
            hypothesis_ids=_string_tuple(
                payload.get("hypothesis_ids", ()), field_name="hypothesis_ids"
            ),
            candidate_run_ids=_string_tuple(
                payload.get("candidate_run_ids", ()), field_name="candidate_run_ids"
            ),
            included_run_ids=_string_tuple(
                payload.get("included_run_ids", ()), field_name="included_run_ids"
            ),
            exclusions=tuple(
                AnalysisExclusion.from_mapping(item)
                for item in _mapping_sequence(payload.get("exclusions", ()), "exclusions")
            ),
            assumptions=tuple(
                AnalysisCheck.from_mapping(item)
                for item in _mapping_sequence(payload.get("assumptions", ()), "assumptions")
            ),
            diagnostics=tuple(
                AnalysisCheck.from_mapping(item)
                for item in _mapping_sequence(payload.get("diagnostics", ()), "diagnostics")
            ),
            tables=_string_tuple(payload.get("tables", ()), field_name="tables"),
            figures=_string_tuple(payload.get("figures", ()), field_name="figures"),
            evidence_refs=_string_tuple(
                payload.get("evidence_refs", ()), field_name="evidence_refs"
            ),
            source_api=str(payload.get("source_api", "")),
            decision_threshold=_optional_float(payload.get("decision_threshold")),
            error=_optional_mapping(payload.get("error"), field_name="error"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation."""
        return {
            "analysis_result_version": self.analysis_result_version,
            "analysis_id": self.analysis_id,
            "analysis_plan_ids": list(self.analysis_plan_ids),
            "hypothesis_ids": list(self.hypothesis_ids),
            "method_id": self.method_id,
            "status": self.status.value,
            "result_type": self.result_type,
            "parameters": _to_jsonable(self.parameters),
            "candidate_run_ids": list(self.candidate_run_ids),
            "included_run_ids": list(self.included_run_ids),
            "exclusions": [item.to_dict() for item in self.exclusions],
            "statistics": _to_jsonable(self.statistics),
            "assumptions": [item.to_dict() for item in self.assumptions],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "tables": list(self.tables),
            "figures": list(self.figures),
            "evidence_refs": list(self.evidence_refs),
            "source_api": self.source_api,
            "decision_threshold": self.decision_threshold,
            "error": None if self.error is None else _to_jsonable(self.error),
        }


@dataclass(frozen=True, slots=True)
class _MethodSupport:
    methods_text: str
    reporting_requirements: tuple[str, ...]
    citation_keys: tuple[str, ...] = ()
    references: tuple[Mapping[str, Any], ...] = ()


def build_analysis_result(
    result: Any,
    *,
    analysis_id: str,
    method_id: str | None = None,
    status: AnalysisStatus | str = AnalysisStatus.COMPLETE,
    analysis_plan_ids: Sequence[str] = (),
    hypothesis_ids: Sequence[str] = (),
    candidate_run_ids: Sequence[str] = (),
    included_run_ids: Sequence[str] = (),
    exclusions: Sequence[AnalysisExclusion | Mapping[str, Any]] = (),
    assumptions: Sequence[AnalysisCheck | Mapping[str, Any]] = (),
    diagnostics: Sequence[AnalysisCheck | Mapping[str, Any]] = (),
    parameters: Mapping[str, Any] | None = None,
    tables: Sequence[str] = (),
    figures: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
    source_api: str = "",
    decision_threshold: float | None = None,
    error: Mapping[str, Any] | None = None,
) -> AnalysisResultRecord:
    """Wrap a built-in result in the portable, paper-oriented result contract."""
    resolved_status = _coerce_status(status)
    if resolved_status == AnalysisStatus.COMPLETE and result is None:
        raise ValueError("A complete analysis requires a result object or mapping.")

    result_type, statistics, inferred_method, inferred_parameters = _normalize_result(result)
    resolved_method = method_id or inferred_method
    if not resolved_method:
        raise ValueError("method_id is required for an unrecognized result type.")
    merged_parameters = dict(inferred_parameters)
    merged_parameters.update(dict(parameters or {}))

    return AnalysisResultRecord(
        analysis_id=analysis_id,
        method_id=resolved_method,
        status=resolved_status,
        result_type=result_type,
        statistics=statistics if resolved_status != AnalysisStatus.PLANNED else {},
        parameters=merged_parameters,
        analysis_plan_ids=_string_tuple(analysis_plan_ids, field_name="analysis_plan_ids"),
        hypothesis_ids=_string_tuple(hypothesis_ids, field_name="hypothesis_ids"),
        candidate_run_ids=_string_tuple(candidate_run_ids, field_name="candidate_run_ids"),
        included_run_ids=_string_tuple(included_run_ids, field_name="included_run_ids"),
        exclusions=tuple(_coerce_exclusion(item) for item in exclusions),
        assumptions=tuple(_coerce_check(item) for item in assumptions),
        diagnostics=tuple(_coerce_check(item) for item in diagnostics),
        tables=_string_tuple(tables, field_name="tables"),
        figures=_string_tuple(figures, field_name="figures"),
        evidence_refs=_string_tuple(evidence_refs, field_name="evidence_refs"),
        source_api=source_api,
        decision_threshold=decision_threshold,
        error=error,
    )


def write_analysis_result(
    result: AnalysisResultRecord | Mapping[str, Any],
    *,
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Atomically write one result beneath ``artifacts/analysis/results``."""
    record = _coerce_record(result)
    destination = (
        Path(output_dir) / "artifacts" / "analysis" / "results" / f"{record.analysis_id}.json"
    )
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Analysis result already exists: {destination}. Pass overwrite=True."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(destination, record.to_dict())
    return destination


def load_analysis_result(path: str | Path) -> AnalysisResultRecord:
    """Load and validate one analysis result in a fresh process."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Analysis result JSON must contain an object.")
    return AnalysisResultRecord.from_mapping(payload)


def collect_analysis_paper_contributions(
    result: AnalysisResultRecord | Mapping[str, Any],
) -> dict[str, Any]:
    """Translate one result into an Experiments-compatible contribution packet."""
    record = _coerce_record(result)
    support = _method_support(record.method_id)
    source = {
        "package": "design-research-analysis",
        "package_version": __version__,
        "component_type": "analysis",
        "component_id": record.method_id,
    }
    evidence_basis = "analyzed" if record.status == AnalysisStatus.COMPLETE else "configured"
    contribution_refs = list(record.evidence_refs)
    contributions: list[dict[str, Any]] = [
        {
            "contribution_id": f"analysis:{record.analysis_id}:methods",
            "section": "methods",
            "kind": "paragraph",
            "text": support.methods_text,
            "evidence_basis": evidence_basis,
            "citation_keys": list(support.citation_keys),
            "evidence_refs": contribution_refs if evidence_basis == "analyzed" else [],
            "metadata": {
                "analysis_id": record.analysis_id,
                "analysis_plan_ids": list(record.analysis_plan_ids),
                "hypothesis_ids": list(record.hypothesis_ids),
                "parameters": _to_jsonable(record.parameters),
                "reporting_requirements": list(support.reporting_requirements),
                "source_api": record.source_api,
            },
        }
    ]
    gaps: list[dict[str, Any]] = []

    if record.status == AnalysisStatus.COMPLETE:
        rendered_results = _render_results(record)
        if not rendered_results:
            gaps.append(
                _gap(
                    record,
                    "unsupported-result-renderer",
                    "results",
                    "The analysis completed, but this result type has no curated Results renderer.",
                )
            )
        else:
            contributions.extend(rendered_results)
        contributions.extend(_artifact_contributions(record, kind="table", paths=record.tables))
        contributions.extend(_artifact_contributions(record, kind="figure", paths=record.figures))
    elif record.status == AnalysisStatus.FAILED:
        error_message = str((record.error or {}).get("message", "No failure message recorded."))
        gaps.append(
            _gap(
                record,
                "analysis-failed",
                "results",
                f"Analysis {record.analysis_id!r} failed: {error_message}",
            )
        )
    else:
        gaps.append(
            _gap(
                record,
                "analysis-not-executed",
                "results",
                f"Analysis {record.analysis_id!r} was planned but has no executed result.",
            )
        )

    gaps.extend(_check_gaps(record, "assumption", record.assumptions))
    gaps.extend(_check_gaps(record, "diagnostic", record.diagnostics))
    gaps.extend(_method_specific_gaps(record))
    if record.candidate_run_ids:
        accounted = len(record.included_run_ids) + len(record.exclusions)
        if accounted != len(record.candidate_run_ids):
            gaps.append(
                _gap(
                    record,
                    "unaccounted-candidate-runs",
                    "results",
                    f"Account for {len(record.candidate_run_ids) - accounted} candidate run(s) "
                    "that are neither included nor excluded.",
                )
            )

    return {
        "schema_version": PAPER_CONTRIBUTION_VERSION,
        "source": source,
        "contributions": contributions,
        "references": [_to_jsonable(item) for item in support.references],
        "reporting_gaps": gaps,
    }


def _normalize_result(result: Any) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Normalize supported result types without mutating their public contracts."""
    if result is None:
        return "None", {}, "", {}
    if isinstance(result, Mapping):
        payload = _to_jsonable(result)
        assert isinstance(payload, dict)
        result_type = "DatasetProfile" if _looks_like_dataset_profile(payload) else "Mapping"
        method_id = "analysis.dataset.profile" if result_type == "DatasetProfile" else ""
    elif hasattr(result, "to_dict"):
        raw_payload = result.to_dict()
        if not isinstance(raw_payload, Mapping):
            raise TypeError("Result to_dict() must return a mapping.")
        payload = _to_jsonable(raw_payload)
        assert isinstance(payload, dict)
        result_type = type(result).__name__
        method_id = _infer_method_id(result)
    else:
        raise TypeError("Analysis results must be supported result objects or mappings.")

    raw_config = payload.pop("config", {})
    parameters = dict(raw_config) if isinstance(raw_config, Mapping) else {"config": raw_config}
    return result_type, payload, method_id, parameters


def _infer_method_id(result: Any) -> str:
    if isinstance(result, RegressionResult):
        return "analysis.stats.ols"
    if isinstance(result, GroupComparisonResult):
        return f"analysis.stats.group-comparison.{result.method}"
    if isinstance(result, ConditionComparisonReport):
        return "analysis.stats.condition-comparison.permutation"
    if isinstance(result, InterraterReliabilityResult):
        return f"analysis.stats.interrater.{result.method.replace('_', '-')}"
    if isinstance(result, MixedEffectsResult):
        return "analysis.stats.mixed-effects"
    if isinstance(result, MarkovChainResult):
        return "analysis.sequence.markov-chain"
    if isinstance(result, ComparisonResult):
        if result.metric == "transition_profile" or "MarkovChain" in {
            result.left_type,
            result.right_type,
        }:
            return "analysis.sequence.markov-comparison"
        return "analysis.result-comparison"
    if isinstance(result, EmbeddingMapResult):
        return f"analysis.embedding-map.{result.method.lower()}"
    if isinstance(result, LanguageConvergenceResult):
        return "analysis.language.convergence"
    return ""


def _method_support(method_id: str) -> _MethodSupport:
    """Return curated method wording and reporting requirements near the implementation."""
    if method_id == "analysis.stats.ols":
        return _MethodSupport(
            methods_text=(
                "An ordinary least-squares regression was fit with the recorded outcome, "
                "predictors, feature encoding, and intercept setting."
            ),
            reporting_requirements=(
                "Report outcome and predictor definitions.",
                "Report sample size, coefficients, model fit, and diagnostic checks.",
                "Do not infer coefficient uncertainty from point estimates alone.",
            ),
        )
    if method_id.startswith("analysis.stats.group-comparison"):
        return _MethodSupport(
            methods_text=(
                "The recorded groups were compared with the selected two-group or omnibus "
                "test, retaining group sizes, means, the test statistic, p-value, and effect size."
            ),
            reporting_requirements=(
                "Name the test and alternative hypothesis.",
                "Report group sizes, descriptive statistics, effect size, and p-value.",
            ),
        )
    if method_id == "analysis.stats.condition-comparison.permutation":
        return _MethodSupport(
            methods_text=(
                "Condition pairs were compared with the recorded exact or seeded sampled "
                "permutation procedure, alternative hypothesis, and decision threshold."
            ),
            reporting_requirements=(
                "Report whether enumeration was exact or sampled.",
                "Report permutations evaluated, effect size, p-value, and prespecified alpha.",
            ),
        )
    if method_id.startswith("analysis.stats.interrater"):
        return _reliability_support(method_id)
    if method_id == "analysis.stats.mixed-effects":
        return _MethodSupport(
            methods_text=(
                "A mixed-effects model was fit with the recorded formula, grouping variable, "
                "estimation setting, and software backend."
            ),
            reporting_requirements=(
                "Report the formula, grouping structure, estimation method, convergence, and fit.",
            ),
        )
    if method_id == "analysis.sequence.markov-chain":
        return _MethodSupport(
            methods_text=(
                "A discrete Markov chain was estimated from the recorded event sequences using "
                "the specified order and additive-smoothing parameter."
            ),
            reporting_requirements=(
                "Report state construction, chain order, smoothing, and sequence count.",
            ),
        )
    if method_id == "analysis.sequence.markov-comparison":
        return _MethodSupport(
            methods_text=(
                "Two fitted Markov chains were aligned by state labels and compared over their "
                "transition profiles with the recorded comparison statistic."
            ),
            reporting_requirements=(
                "Report state alignment, comparison metric, estimate, and uncertainty if tested.",
            ),
        )
    if method_id.startswith("analysis.embedding-map"):
        return _embedding_support(method_id)
    if method_id == "analysis.language.convergence":
        return _MethodSupport(
            methods_text=(
                "Language convergence was evaluated from within-group semantic-distance "
                "trajectories using the recorded embedding provider and window size."
            ),
            reporting_requirements=(
                "Report the text source, embedding method, grouping, window size, and slope rule.",
            ),
        )
    if method_id == "analysis.dataset.profile":
        return _MethodSupport(
            methods_text=(
                "The analysis dataset was profiled for row and column counts, missingness, "
                "numeric summaries, categorical cardinality, and IQR-based outlier flags."
            ),
            reporting_requirements=(
                "Report dataset dimensions, missingness, exclusions, and material profile "
                "warnings.",
            ),
        )
    return _MethodSupport(
        methods_text=(
            f"The analysis used the recorded method identifier {method_id!r}; no more specific "
            "curated method description is registered."
        ),
        reporting_requirements=("Provide a method-specific description and reporting checklist.",),
    )


def _reliability_support(method_id: str) -> _MethodSupport:
    references = {
        "analysis.stats.interrater.cohen-kappa": (
            "cohen1960",
            {
                "key": "cohen1960",
                "title": "A coefficient of agreement for nominal scales",
                "authors": ["Jacob Cohen"],
                "year": 1960,
                "doi": "10.1177/001316446002000104",
                "raw_text": (
                    "@article{cohen1960, author={Cohen, Jacob}, "
                    "title={A Coefficient of Agreement for Nominal Scales}, "
                    "journal={Educational and Psychological Measurement}, year={1960}, "
                    "volume={20}, number={1}, pages={37--46}, "
                    "doi={10.1177/001316446002000104}}"
                ),
            },
        ),
        "analysis.stats.interrater.fleiss-kappa": (
            "fleiss1971",
            {
                "key": "fleiss1971",
                "title": "Measuring nominal scale agreement among many raters",
                "authors": ["Joseph L. Fleiss"],
                "year": 1971,
                "doi": "10.1037/h0031619",
                "raw_text": (
                    "@article{fleiss1971, author={Fleiss, Joseph L.}, "
                    "title={Measuring Nominal Scale Agreement among Many Raters}, "
                    "journal={Psychological Bulletin}, year={1971}, volume={76}, "
                    "number={5}, pages={378--382}, doi={10.1037/h0031619}}"
                ),
            },
        ),
        "analysis.stats.interrater.krippendorff-alpha": (
            "krippendorff2018",
            {
                "key": "krippendorff2018",
                "title": "Content Analysis: An Introduction to Its Methodology",
                "authors": ["Klaus Krippendorff"],
                "year": 2018,
                "raw_text": (
                    "@book{krippendorff2018, author={Krippendorff, Klaus}, "
                    "title={Content Analysis: An Introduction to Its Methodology}, "
                    "edition={4}, publisher={SAGE Publications}, year={2018}}"
                ),
            },
        ),
    }
    citation_key, reference = references.get(method_id, ("", {}))
    label = method_id.rsplit(".", maxsplit=1)[-1].replace("-", " ")
    return _MethodSupport(
        methods_text=(
            f"Inter-rater reliability was estimated with {label}, retaining the number of "
            "items, raters, usable observations, missing ratings, and bootstrap interval settings."
        ),
        reporting_requirements=(
            "Report the coefficient, item and rater counts, missing-data rule, and interval "
            "method.",
        ),
        citation_keys=(citation_key,) if citation_key else (),
        references=(reference,) if reference else (),
    )


def _embedding_support(method_id: str) -> _MethodSupport:
    method = method_id.rsplit(".", maxsplit=1)[-1]
    references: tuple[Mapping[str, Any], ...] = ()
    citation_keys: tuple[str, ...] = ()
    if method == "umap":
        citation_keys = ("mcinnes2018umap",)
        references = (
            {
                "key": "mcinnes2018umap",
                "title": "UMAP: Uniform Manifold Approximation and Projection",
                "authors": ["Leland McInnes", "John Healy", "James Melville"],
                "year": 2018,
                "doi": "10.21105/joss.00861",
                "raw_text": (
                    "@article{mcinnes2018umap, author={McInnes, Leland and Healy, John and "
                    "Melville, James}, title={UMAP: Uniform Manifold Approximation and "
                    "Projection for Dimension Reduction}, journal={Journal of Open Source "
                    "Software}, year={2018}, volume={3}, number={29}, pages={861}, "
                    "doi={10.21105/joss.00861}}"
                ),
            },
        )
    return _MethodSupport(
        methods_text=(
            f"Records were projected with the {method.upper()} embedding-map method using the "
            "recorded component count, random seed, and method-specific parameters."
        ),
        reporting_requirements=(
            "Report input representation, projection parameters, random seed, and limitations.",
        ),
        citation_keys=citation_keys,
        references=references,
    )


def _render_results(record: AnalysisResultRecord) -> list[dict[str, Any]]:
    payload = record.statistics
    if record.result_type == "RegressionResult":
        coefficients = _format_mapping(payload.get("coefficients", {}))
        text = (
            f"The regression used {payload.get('n_samples')} samples and "
            f"{payload.get('n_features')} predictors (R²={_format_number(payload.get('r2'))}, "
            f"MSE={_format_number(payload.get('mse'))}); fitted coefficients were {coefficients}."
        )
        return [_result_contribution(record, "summary", text)]
    if record.result_type == "GroupComparisonResult":
        text = (
            f"The {payload.get('method')} comparison returned "
            f"{_test_statistic_sentence(payload, threshold=record.decision_threshold)} "
            f"Group means were {_format_mapping(payload.get('group_means', {}))}; group sizes "
            f"were {_format_mapping(payload.get('group_sizes', {}), integer=True)}."
        )
        return [_result_contribution(record, "summary", text)]
    if record.result_type == "ConditionComparisonReport":
        rows = payload.get("comparisons", [])
        if not isinstance(rows, list):
            return []
        return [
            _result_contribution(
                record,
                f"pair-{index + 1}",
                (
                    f"For {item.get('pair_label', 'the condition pair')}, the recorded mean "
                    f"difference was {_format_number(item.get('mean_difference'))}, Cohen's "
                    f"d={_format_number(item.get('effect_size'))}, and "
                    f"p={_format_number(item.get('p_value'))} "
                    f"({item.get('test_method')} permutation test)."
                ),
            )
            for index, item in enumerate(rows)
            if isinstance(item, Mapping)
        ]
    if record.result_type == "InterraterReliabilityResult":
        interval = payload.get("confidence_interval")
        interval_text = ""
        if isinstance(interval, list) and len(interval) == 2:
            interval_text = (
                f", with interval [{_format_number(interval[0])}, {_format_number(interval[1])}]"
            )
        text = (
            f"{str(payload.get('method', '')).replace('_', ' ').title()} was "
            f"{_format_number(payload.get('coefficient'))}{interval_text}, using "
            f"{payload.get('n_items_used')} of {payload.get('n_items')} items, "
            f"{payload.get('n_raters')} raters, and {payload.get('n_observations')} ratings; "
            f"{payload.get('missing_ratings')} ratings were missing."
        )
        return [_result_contribution(record, "summary", text)]
    if record.result_type == "ComparisonResult":
        text = (
            f"The {payload.get('metric')} comparison estimate was "
            f"{_format_number(payload.get('estimate'))}"
            f"{_optional_statistic_tail(payload, threshold=record.decision_threshold)}."
        )
        return [_result_contribution(record, "summary", text)]
    if record.result_type == "MarkovChainResult":
        states = payload.get("states", [])
        text = (
            f"The fitted order-{payload.get('order')} Markov chain contained "
            f"{len(states) if isinstance(states, list) else 'an unrecorded number of'} states "
            f"and used additive smoothing {_format_number(payload.get('smoothing'))}."
        )
        return [_result_contribution(record, "summary", text)]
    if record.result_type == "MixedEffectsResult":
        if not payload.get("success"):
            return []
        text = (
            f"The mixed-effects model fit successfully with log likelihood "
            f"{_format_number(payload.get('log_likelihood'))}, AIC "
            f"{_format_number(payload.get('aic'))}, and BIC {_format_number(payload.get('bic'))}; "
            f"estimated parameters were {_format_mapping(payload.get('params', {}))}."
        )
        return [_result_contribution(record, "summary", text)]
    if record.result_type == "DatasetProfile":
        warnings = payload.get("warnings", [])
        warning_text = (
            f" The profiler recorded {len(warnings)} warning(s)."
            if isinstance(warnings, list)
            else ""
        )
        text = (
            f"The analyzed dataset contained {payload.get('n_rows')} rows and "
            f"{payload.get('n_columns')} columns.{warning_text}"
        )
        return [_result_contribution(record, "summary", text)]
    if record.result_type == "EmbeddingMapResult":
        shape = payload.get("shape", [])
        record_count = shape[0] if isinstance(shape, list) and shape else "an unrecorded number of"
        dimension_count = (
            shape[1] if isinstance(shape, list) and len(shape) > 1 else "an unrecorded number of"
        )
        text = (
            f"The {payload.get('method')} projection represented "
            f"{record_count} records in {dimension_count} dimensions."
        )
        return [_result_contribution(record, "summary", text)]
    if record.result_type == "LanguageConvergenceResult":
        text = (
            f"Language convergence was evaluated for {len(payload.get('groups', []))} group(s) "
            f"across {payload.get('n_observations')} observations; recorded slope directions were "
            f"{_format_mapping(payload.get('direction_by_group', {}))}."
        )
        return [_result_contribution(record, "summary", text)]
    return []


def _result_contribution(
    record: AnalysisResultRecord,
    suffix: str,
    text: str,
) -> dict[str, Any]:
    return {
        "contribution_id": f"analysis:{record.analysis_id}:results:{suffix}",
        "section": "results",
        "kind": "paragraph" if suffix == "summary" else "bullet",
        "text": text,
        "evidence_basis": "analyzed",
        "citation_keys": [],
        "evidence_refs": list(record.evidence_refs),
        "metadata": {
            "analysis_id": record.analysis_id,
            "analysis_plan_ids": list(record.analysis_plan_ids),
            "hypothesis_ids": list(record.hypothesis_ids),
            "included_run_ids": list(record.included_run_ids),
            "exclusions": [item.to_dict() for item in record.exclusions],
        },
    }


def _artifact_contributions(
    record: AnalysisResultRecord,
    *,
    kind: str,
    paths: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "contribution_id": f"analysis:{record.analysis_id}:{kind}:{index + 1}",
            "section": "results",
            "kind": kind,
            "text": f"{kind.title()} generated by analysis {record.analysis_id!r}: `{path}`.",
            "evidence_basis": "analyzed",
            "citation_keys": [],
            "evidence_refs": list(record.evidence_refs),
            "metadata": {"analysis_id": record.analysis_id, "path": path},
        }
        for index, path in enumerate(paths)
    ]


def _check_gaps(
    record: AnalysisResultRecord,
    kind: str,
    checks: Sequence[AnalysisCheck],
) -> list[dict[str, Any]]:
    return [
        _gap(
            record,
            f"{kind}-{check.check_id}",
            "methods" if kind == "assumption" else "results",
            f"{kind.title()} check {check.check_id!r} failed: {check.detail}",
            evidence_refs=record.evidence_refs,
        )
        for check in checks
        if check.status == "failed"
    ]


def _method_specific_gaps(record: AnalysisResultRecord) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    if record.status != AnalysisStatus.COMPLETE:
        return gaps
    if record.method_id == "analysis.stats.ols" and not any(
        key in record.statistics for key in ("standard_errors", "p_values")
    ):
        gaps.append(
            _gap(
                record,
                "regression-uncertainty",
                "results",
                "The OLS result records point estimates and fit but not coefficient "
                "uncertainty; add an inferential result before making significance claims.",
                evidence_refs=record.evidence_refs,
            )
        )
    if record.result_type == "InterraterReliabilityResult" and not record.statistics.get(
        "confidence_interval"
    ):
        gaps.append(
            _gap(
                record,
                "reliability-interval",
                "results",
                "No uncertainty interval was recorded for the reliability coefficient.",
                evidence_refs=record.evidence_refs,
            )
        )
    if record.result_type == "MixedEffectsResult" and not record.statistics.get("success"):
        gaps.append(
            _gap(
                record,
                "mixed-effects-fit-failed",
                "results",
                "The mixed-effects backend returned an unsuccessful fit; do not report estimates.",
                evidence_refs=record.evidence_refs,
            )
        )
    if record.method_id.startswith("analysis.embedding-map") and not record.figures:
        gaps.append(
            _gap(
                record,
                "embedding-map-figure",
                "results",
                "No rendered embedding-map figure was attached to the completed analysis.",
                evidence_refs=record.evidence_refs,
            )
        )
    return gaps


def _gap(
    record: AnalysisResultRecord,
    suffix: str,
    section: str,
    message: str,
    *,
    evidence_refs: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "gap_id": f"analysis:{record.analysis_id}:{suffix}",
        "section": section,
        "message": message,
        "evidence_refs": list(evidence_refs),
    }


def _test_statistic_sentence(
    payload: Mapping[str, Any],
    *,
    threshold: float | None,
) -> str:
    text = (
        f"statistic={_format_number(payload.get('statistic'))}, "
        f"p={_format_number(payload.get('p_value'))}, and effect "
        f"size={_format_number(payload.get('effect_size'))}."
    )
    p_value = payload.get("p_value")
    if threshold is not None and isinstance(p_value, int | float):
        outcome = "met" if float(p_value) < threshold else "did not meet"
        text += f" The result {outcome} the prespecified alpha={threshold:g} threshold."
    return text


def _optional_statistic_tail(
    payload: Mapping[str, Any],
    *,
    threshold: float | None,
) -> str:
    parts: list[str] = []
    for key, label in (
        ("statistic", "statistic"),
        ("p_value", "p"),
        ("effect_size", "effect size"),
    ):
        value = payload.get(key)
        if value is not None:
            parts.append(f"{label}={_format_number(value)}")
    tail = f" ({', '.join(parts)})" if parts else ""
    p_value = payload.get("p_value")
    if threshold is not None and isinstance(p_value, int | float):
        outcome = "met" if float(p_value) < threshold else "did not meet"
        tail += f", which {outcome} the prespecified alpha={threshold:g} threshold"
    return tail


def _format_mapping(value: Any, *, integer: bool = False) -> str:
    if not isinstance(value, Mapping) or not value:
        return "not recorded"
    return ", ".join(
        f"{key}={int(item) if integer and isinstance(item, int | float) else _format_number(item)}"
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    )


def _format_number(value: Any) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.4g}"
    return str(value)


def _looks_like_dataset_profile(payload: Mapping[str, Any]) -> bool:
    return {"n_rows", "n_columns", "columns", "warnings"}.issubset(payload)


def _coerce_record(
    result: AnalysisResultRecord | Mapping[str, Any],
) -> AnalysisResultRecord:
    if isinstance(result, AnalysisResultRecord):
        return result
    if not isinstance(result, Mapping):
        raise TypeError("Analysis result must be an AnalysisResultRecord or mapping.")
    return AnalysisResultRecord.from_mapping(result)


def _coerce_exclusion(item: AnalysisExclusion | Mapping[str, Any]) -> AnalysisExclusion:
    if isinstance(item, AnalysisExclusion):
        return item
    return AnalysisExclusion.from_mapping(item)


def _coerce_check(item: AnalysisCheck | Mapping[str, Any]) -> AnalysisCheck:
    if isinstance(item, AnalysisCheck):
        return item
    return AnalysisCheck.from_mapping(item)


def _coerce_status(value: Any) -> AnalysisStatus:
    try:
        return AnalysisStatus(str(value))
    except ValueError as exc:
        raise ValueError("status must be one of: planned, complete, failed.") from exc


def _required_text(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} {key} must be non-empty text.")
    return value


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object.")
    return dict(value)


def _optional_mapping(value: Any, *, field_name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _mapping(value, field_name=field_name)


def _mapping_sequence(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence of JSON objects.")
    items: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name} must contain only JSON objects.")
        items.append(item)
    return tuple(items)


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence of strings.")
    result = tuple(str(item) for item in value)
    if any(not item.strip() for item in result):
        raise ValueError(f"{field_name} cannot contain blank values.")
    return result


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _require_unique(values: Sequence[str], *, field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates.")


def _validate_relative_artifact_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.strip() in {"", "."}:
        raise ValueError("Analysis table and figure paths must be safe relative paths.")


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


__all__ = [
    "ANALYSIS_RESULT_VERSION",
    "PAPER_CONTRIBUTION_VERSION",
    "AnalysisCheck",
    "AnalysisExclusion",
    "AnalysisResultRecord",
    "AnalysisStatus",
    "build_analysis_result",
    "collect_analysis_paper_contributions",
    "load_analysis_result",
    "write_analysis_result",
]
