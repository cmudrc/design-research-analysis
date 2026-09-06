"""Create and verify a narrow paper-draft research bundle.

## Introduction
Package a completed study handoff for collaborator inspection, manual
reanalysis, and continued paper editing without promising automatic replay or
environment recreation.

## Technical Implementation
1. Write a tiny canonical study fixture and one retained run record.
2. Persist one structured analysis result and a complete paper-draft tree.
3. Explicitly select one supporting-data table while leaving an attachment out.
4. Create the deterministic ZIP and verify every member hash without extraction.

## Expected Results
Prints a successful verification summary for ``study-paper-draft.zip``. The
archive contains canonical artifacts, run and analysis records, the paper
draft, sanitized environment metadata, and the explicitly selected support
file; the unselected attachment is absent.

## References
- docs/research_bundle.rst
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import design_research_analysis as dran


def _write_text(path: Path, content: str) -> None:
    """Write one fixture file after creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_study_fixture(root: Path) -> None:
    """Write the minimum verified study, evidence, analysis, and draft tree."""
    _write_text(
        root / "study.yaml",
        "schema_version: 0.2.0\nstudy_id: bundle-example\n"
        "title: Bundle example\ndescription: Deterministic fixture\n",
    )
    _write_text(
        root / "manifest.json",
        json.dumps({"schema_version": "0.2.0", "study_id": "bundle-example"}),
    )
    for filename in ("conditions.csv", "runs.csv", "events.csv", "evaluations.csv"):
        _write_text(root / filename, "id\n1\n")
    _write_text(root / "hypotheses.json", "[]\n")
    _write_text(root / "analysis_plan.json", "[]\n")

    evidence_ref = "artifacts/runs/run-1/run.json"
    _write_text(
        root / evidence_ref,
        json.dumps({"schema_version": "0.1.0", "run_id": "run-1", "status": "success"}),
    )
    _write_text(root / "artifacts/runs/run-1/observations.jsonl", '{"event": "observed"}\n')
    _write_text(root / "artifacts/runs/run-1/attachments/private.txt", "not selected\n")

    table_ref = "artifacts/analysis/tables/result.tex"
    _write_text(root / table_ref, "\\begin{tabular}{lr}A & 1\\\\\\end{tabular}\n")
    (root / "artifacts/analysis/figures").mkdir(parents=True)
    result = dran.build_analysis_result(
        {"estimate": 1.0},
        analysis_id="bundle-example-analysis",
        method_id="analysis.example",
        candidate_run_ids=("run-1",),
        included_run_ids=("run-1",),
        tables=(table_ref,),
        evidence_refs=(evidence_ref,),
    )
    dran.write_analysis_result(result, output_dir=root)

    _write_text(root / "paper-draft/main.tex", "\\documentclass{article}\n")
    _write_text(root / "paper-draft/paper_draft.md", "# Paper draft\n")
    _write_text(root / "paper-draft/references.bib", "")
    _write_text(root / "paper-draft/README.md", "Author review required.\n")
    _write_text(
        root / "paper-draft/paper_draft_manifest.json",
        json.dumps({"document_status": "paper-draft", "study_id": "bundle-example"}),
    )
    for section in ("introduction", "background", "methods", "results", "discussion"):
        _write_text(
            root / "paper-draft/sections" / f"{section}.tex",
            f"\\section{{{section.title()}}}\n",
        )
    (root / "paper-draft/tables").mkdir()
    (root / "paper-draft/figures").mkdir()
    _write_text(
        root / "artifacts/analysis/supporting-data/selected.csv",
        "run_id,value\nrun-1,1\n",
    )


def main() -> None:
    """Create and verify the example bundle in a temporary directory."""
    with TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir) / "study-output"
        _write_study_fixture(root)
        bundle = dran.create_research_bundle(
            root,
            supporting_data=("artifacts/analysis/supporting-data/selected.csv",),
        )
        verification = dran.verify_research_bundle(bundle)
        assert verification["bundle_schema_version"] == dran.BUNDLE_SCHEMA_VERSION
        print(f"Bundle: {bundle.name}")
        print(f"Verified: {verification['valid']}")
        print(f"Members: {verification['file_count']}")


if __name__ == "__main__":
    main()
