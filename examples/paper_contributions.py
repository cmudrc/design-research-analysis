"""Export evidence-linked analysis results and paper contributions.

## Introduction
Wrap an existing analysis result in a portable record that distinguishes executed
evidence from planned methods and produces restrained Methods and Results support.

## Technical Implementation
1. Fit a deterministic ordinary least-squares regression.
2. Record the hypothesis, included runs, documented exclusion, and diagnostic checks.
3. Write and reload the result beneath the study artifact directory.
4. Collect a JSON-compatible packet for downstream paper-draft assembly.

## Expected Results
Prints the two contract versions, the reloaded analysis identity, and the generated
contribution and reporting-gap identifiers. The gap makes the missing coefficient
uncertainty visible instead of implying a significance test.

## References
- docs/paper_contributions.rst
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import design_research_analysis as dran


def main() -> None:
    """Build, persist, reload, and translate one analysis result."""
    regression = dran.fit_regression(
        [[0.0], [1.0], [2.0], [3.0]],
        [1.0, 1.5, 2.0, 2.5],
        feature_names=["iteration"],
    )
    record = dran.build_analysis_result(
        regression,
        analysis_id="h1-iteration-regression",
        status=dran.AnalysisStatus.COMPLETE,
        analysis_plan_ids=("plan-h1",),
        hypothesis_ids=("H1",),
        candidate_run_ids=("run-1", "run-2", "run-3", "run-4"),
        included_run_ids=("run-1", "run-2", "run-3"),
        exclusions=(
            dran.AnalysisExclusion(
                run_id="run-4",
                reason="The prespecified evaluator output was missing.",
            ),
        ),
        assumptions=(
            dran.AnalysisCheck(
                check_id="linearity",
                status="passed",
                detail="The deterministic fixture is exactly linear.",
            ),
        ),
        evidence_refs=(
            "artifacts/runs/run-1/run.json",
            "artifacts/runs/run-2/run.json",
            "artifacts/runs/run-3/run.json",
        ),
        source_api="fit_regression",
    )

    with TemporaryDirectory() as temporary_dir:
        result_path = dran.write_analysis_result(
            record,
            output_dir=Path(temporary_dir),
        )
        reloaded = dran.load_analysis_result(result_path)
        assert isinstance(reloaded, dran.AnalysisResultRecord)
        packet = dran.collect_analysis_paper_contributions(reloaded)

    assert reloaded.status is dran.AnalysisStatus.COMPLETE
    assert reloaded.analysis_result_version == dran.ANALYSIS_RESULT_VERSION
    assert packet["schema_version"] == dran.PAPER_CONTRIBUTION_VERSION
    print("Analysis result contract:", dran.ANALYSIS_RESULT_VERSION)
    print("Paper contribution contract:", dran.PAPER_CONTRIBUTION_VERSION)
    print("Reloaded analysis:", reloaded.analysis_id)
    print(
        "Contributions:",
        ", ".join(item["contribution_id"] for item in packet["contributions"]),
    )
    print(
        "Reporting gaps:",
        ", ".join(item["gap_id"] for item in packet["reporting_gaps"]),
    )


if __name__ == "__main__":
    main()
