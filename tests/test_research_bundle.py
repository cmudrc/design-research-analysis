"""Tests for verified paper-draft research bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from design_research_analysis import (
    BUNDLE_SCHEMA_VERSION,
    build_analysis_result,
    create_research_bundle,
    verify_research_bundle,
    write_analysis_result,
)


def _write_fixture(tmp_path: Path) -> Path:
    """Write a minimal complete study, analysis, and paper-draft tree."""
    root = tmp_path / "study-output"
    root.mkdir(parents=True)
    (root / "study.yaml").write_text(
        "schema_version: 0.2.0\nstudy_id: bundle-study\ntitle: Bundle study\ndescription: Test\n",
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": "0.2.0", "study_id": "bundle-study"}),
        encoding="utf-8",
    )
    for filename in ("conditions.csv", "runs.csv", "events.csv", "evaluations.csv"):
        (root / filename).write_text("id\n1\n", encoding="utf-8")
    (root / "hypotheses.json").write_text("[]\n", encoding="utf-8")
    (root / "analysis_plan.json").write_text("[]\n", encoding="utf-8")
    (root / "component_metadata.json").write_text('{"packets": []}\n', encoding="utf-8")
    (root / "analysis_results.json").write_text('{"results": []}\n', encoding="utf-8")

    run_dir = root / "artifacts" / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"schema_version": "0.1.0", "run_id": "run-1", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "observations.jsonl").write_text('{"event": "observed"}\n', encoding="utf-8")
    attachments = run_dir / "attachments"
    attachments.mkdir()
    (attachments / "participant-secret.txt").write_text("excluded", encoding="utf-8")

    table = root / "artifacts" / "analysis" / "tables" / "h1.tex"
    table.parent.mkdir(parents=True)
    table.write_text("\\begin{tabular}{lr}A & 1\\\\\\end{tabular}\n", encoding="utf-8")
    (root / "artifacts" / "analysis" / "figures").mkdir()
    record = build_analysis_result(
        {"estimate": 1.0},
        analysis_id="h1-analysis",
        method_id="analysis.test.method",
        candidate_run_ids=("run-1",),
        included_run_ids=("run-1",),
        tables=("artifacts/analysis/tables/h1.tex",),
        evidence_refs=("artifacts/runs/run-1/run.json",),
    )
    write_analysis_result(record, output_dir=root)

    draft = root / "paper-draft"
    (draft / "sections").mkdir(parents=True)
    (draft / "tables").mkdir()
    (draft / "figures").mkdir()
    (draft / "main.tex").write_text("\\documentclass{article}\n", encoding="utf-8")
    (draft / "paper_draft.md").write_text("# Paper draft\n", encoding="utf-8")
    (draft / "references.bib").write_text("", encoding="utf-8")
    (draft / "README.md").write_text("Author review required.\n", encoding="utf-8")
    (draft / "paper_draft_manifest.json").write_text(
        json.dumps(
            {
                "paper_draft_version": "0.1.0",
                "document_status": "paper-draft",
                "study_id": "bundle-study",
            }
        ),
        encoding="utf-8",
    )
    for section in ("introduction", "background", "methods", "results", "discussion"):
        (draft / "sections" / f"{section}.tex").write_text(
            f"\\section{{{section.title()}}}\n",
            encoding="utf-8",
        )

    supporting = root / "artifacts" / "analysis" / "supporting-data"
    supporting.mkdir()
    (supporting / "selected.csv").write_text("run_id,value\nrun-1,1\n", encoding="utf-8")
    (root / "unselected-participant-data.csv").write_text("secret\n", encoding="utf-8")
    return root


def test_bundle_is_deterministic_sanitized_complete_and_verifiable(tmp_path: Path) -> None:
    """The bundle should be portable, deterministic, narrow, and integrity checked."""
    root = _write_fixture(tmp_path)
    selected = "artifacts/analysis/supporting-data/selected.csv"
    first = create_research_bundle(
        root / "manifest.json",
        output_path=tmp_path / "first.zip",
        supporting_data=(selected,),
    )
    second = create_research_bundle(
        root,
        output_path=tmp_path / "second.zip",
        supporting_data=(selected,),
    )
    assert first.read_bytes() == second.read_bytes()

    verification = verify_research_bundle(first)
    assert verification["valid"] is True
    assert verification["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
    assert verification["study_id"] == "bundle-study"
    assert verification["file_count"] > 20

    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        prefix = "study-paper-draft/"
        assert prefix + "manifest.json" in names
        assert prefix + "artifacts/runs/run-1/run.json" in names
        assert prefix + "artifacts/runs/run-1/observations.jsonl" in names
        assert prefix + "artifacts/analysis/results/h1-analysis.json" in names
        assert prefix + "paper-draft/main.tex" in names
        assert prefix + selected in names
        assert prefix + "bundle/environment.json" in names
        assert prefix + "bundle_manifest.json" in names
        assert not any("participant-secret" in name for name in names)
        assert prefix + "unselected-participant-data.csv" not in names
        environment = json.loads(archive.read(prefix + "bundle/environment.json"))
        assert "node" not in json.dumps(environment)
        assert "executable" not in environment
        manifest = json.loads(archive.read(prefix + "bundle_manifest.json"))
        assert manifest["inventory_hash_scope"] == "all members except bundle_manifest.json"
        assert manifest["bundle_manifest_integrity"] == "SHA-256 stored in the ZIP comment"
        assert manifest["selected_supporting_data"] == [selected]
        assert "storage or tracking platform" in manifest["non_goals"]


def test_bundle_refuses_overwrite_and_detects_changed_member(tmp_path: Path) -> None:
    """Replacement must be explicit and member tampering must fail verification."""
    root = _write_fixture(tmp_path)
    bundle = create_research_bundle(root)
    with pytest.raises(FileExistsError, match="already exists"):
        create_research_bundle(root)
    assert create_research_bundle(root, overwrite=True) == bundle

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            content = source.read(info.filename)
            if info.filename.endswith("study.yaml"):
                content = content.replace(b"Bundle study", b"Fundle study")
            target.writestr(info, content)
        target.comment = source.comment
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_research_bundle(tampered)


def test_bundle_rejects_missing_or_unsafe_sources(tmp_path: Path) -> None:
    """Missing evidence, path traversal, symlinks, and broken artifact refs should fail."""
    root = _write_fixture(tmp_path)
    for unsafe in ("../outside.csv", str(tmp_path / "absolute.csv"), "."):
        with pytest.raises(ValueError):
            create_research_bundle(
                root,
                output_path=tmp_path / f"unsafe-{len(unsafe)}.zip",
                supporting_data=(unsafe,),
            )

    outside = tmp_path / "outside.csv"
    outside.write_text("secret", encoding="utf-8")
    link = root / "linked.csv"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        create_research_bundle(
            root,
            output_path=tmp_path / "symlink.zip",
            supporting_data=("linked.csv",),
        )

    (root / "artifacts" / "analysis" / "tables" / "h1.tex").unlink()
    with pytest.raises(ValueError, match="missing artifact"):
        create_research_bundle(root, output_path=tmp_path / "missing-table.zip")

    empty_root = tmp_path / "empty-study"
    empty_root.mkdir()
    with pytest.raises(ValueError, match="Missing required bundle files"):
        create_research_bundle(empty_root)


def test_verifier_rejects_unsafe_or_malformed_archives(tmp_path: Path) -> None:
    """Verification should inspect names and format before any extraction occurs."""
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="unsafe member"):
        verify_research_bundle(unsafe)

    not_zip = tmp_path / "not.zip"
    not_zip.write_text("not a zip", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid research bundle"):
        verify_research_bundle(not_zip)
    with pytest.raises(ValueError, match=r"existing \.zip"):
        verify_research_bundle(tmp_path / "missing.zip")


def _rewrite_bundle_manifest(
    source_path: Path,
    target_path: Path,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    """Rewrite a bundle with a mutated manifest and matching ZIP-comment hash."""
    manifest_name = "study-paper-draft/bundle_manifest.json"
    with zipfile.ZipFile(source_path) as source, zipfile.ZipFile(target_path, "w") as target:
        for info in source.infolist():
            content = source.read(info.filename)
            if info.filename == manifest_name:
                payload = json.loads(content)
                mutate(payload)
                content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
                target.comment = (
                    "bundle-manifest-sha256:" + hashlib.sha256(content).hexdigest()
                ).encode()
            target.writestr(info, content)


def test_creation_validation_covers_incomplete_inputs_and_output_safety(tmp_path: Path) -> None:
    """Creation should fail before publishing incomplete or recursively selected output."""
    root = _write_fixture(tmp_path / "base")
    with pytest.raises(ValueError, match=r"use a \.zip"):
        create_research_bundle(root, output_path=tmp_path / "bundle.tar")
    with pytest.raises(ValueError, match="selected source"):
        create_research_bundle(root, output_path=root / "paper-draft" / "nested.zip")
    output_directory = tmp_path / "directory.zip"
    output_directory.mkdir()
    with pytest.raises(ValueError, match="regular"):
        create_research_bundle(root, output_path=output_directory, overwrite=True)
    with pytest.raises(ValueError, match="does not exist"):
        create_research_bundle(
            root,
            output_path=tmp_path / "missing-support.zip",
            supporting_data=("missing.csv",),
        )

    malformed_manifest = _write_fixture(tmp_path / "manifest")
    (malformed_manifest / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        create_research_bundle(malformed_manifest)

    unsupported = _write_fixture(tmp_path / "schema")
    (unsupported / "manifest.json").write_text(
        json.dumps({"schema_version": "9.9.9", "study_id": "bundle-study"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported experiment artifact schema"):
        create_research_bundle(unsupported)

    no_runs = _write_fixture(tmp_path / "runs")
    shutil.rmtree(no_runs / "artifacts" / "runs")
    with pytest.raises(ValueError, match="at least one per-run"):
        create_research_bundle(no_runs)

    no_results = _write_fixture(tmp_path / "results")
    shutil.rmtree(no_results / "artifacts" / "analysis" / "results")
    with pytest.raises(ValueError, match="at least one analysis-result"):
        create_research_bundle(no_results)

    wrong_draft = _write_fixture(tmp_path / "draft")
    (wrong_draft / "paper-draft" / "paper_draft_manifest.json").write_text(
        json.dumps({"document_status": "final", "study_id": "bundle-study"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="document_status"):
        create_research_bundle(wrong_draft)


def test_verifier_rejects_manifest_inventory_and_member_inconsistencies(tmp_path: Path) -> None:
    """Every level of the two-part integrity envelope should be checked."""
    root = _write_fixture(tmp_path / "valid")
    bundle = create_research_bundle(root, output_path=tmp_path / "valid.zip")

    cases: tuple[tuple[str, Callable[[dict[str, object]], None], str], ...] = (
        (
            "bad-version",
            lambda payload: payload.update(bundle_schema_version="9.9.9"),
            "unsupported bundle schema",
        ),
        (
            "bad-root",
            lambda payload: payload.update(archive_root="other"),
            "unexpected archive root",
        ),
        (
            "bad-study",
            lambda payload: payload.update(study_id=""),
            "missing study_id",
        ),
        (
            "bad-files",
            lambda payload: payload.update(files="not-an-array"),
            "files must be an array",
        ),
        (
            "bad-type",
            lambda payload: payload["files"][0].update(type="other"),  # type: ignore[index,union-attr]
            "invalid type",
        ),
        (
            "bad-size",
            lambda payload: payload["files"][0].update(size_bytes=-1),  # type: ignore[index,union-attr]
            "invalid size",
        ),
        (
            "bad-hash",
            lambda payload: payload["files"][0].update(sha256="short"),  # type: ignore[index,union-attr]
            "invalid hash",
        ),
        (
            "unsafe-path",
            lambda payload: payload["files"][0].update(path="../escape"),  # type: ignore[index,union-attr]
            "unsafe path",
        ),
        (
            "duplicate-path",
            lambda payload: payload["files"].append(payload["files"][0]),  # type: ignore[index,union-attr]
            "repeats path",
        ),
    )
    for name, mutate, message in cases:
        changed = tmp_path / f"{name}.zip"
        _rewrite_bundle_manifest(bundle, changed, mutate)
        with pytest.raises(ValueError, match=message):
            verify_research_bundle(changed)

    extra = tmp_path / "extra.zip"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(extra, "w") as target:
        for info in source.infolist():
            target.writestr(info, source.read(info.filename))
        target.writestr("study-paper-draft/extra.txt", "extra")
        target.comment = source.comment
    with pytest.raises(ValueError, match="members do not match"):
        verify_research_bundle(extra)

    bad_comment = tmp_path / "bad-comment.zip"
    shutil.copyfile(bundle, bad_comment)
    with zipfile.ZipFile(bad_comment, "a") as archive:
        archive.comment = b"missing"
    with pytest.raises(ValueError, match="comment is missing"):
        verify_research_bundle(bad_comment)


def test_additional_creation_and_verification_boundaries(tmp_path: Path) -> None:
    """Validate identity, accounting, directory selection, and integrity mismatches."""
    with pytest.raises(ValueError, match="Expected a study output"):
        create_research_bundle(tmp_path / "missing-study")

    root = _write_fixture(tmp_path / "valid-extra")
    directory_bundle = create_research_bundle(
        root,
        output_path=tmp_path / "support-directory.zip",
        supporting_data=("artifacts/analysis/supporting-data",),
    )
    assert verify_research_bundle(directory_bundle)["valid"] is True

    no_identity = _write_fixture(tmp_path / "identity")
    (no_identity / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must declare study_id"):
        create_research_bundle(no_identity)

    bad_run = _write_fixture(tmp_path / "run-identity")
    (bad_run / "artifacts" / "runs" / "run-1" / "run.json").write_text(
        '{"run_id": "run-1"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must declare run_id and status"):
        create_research_bundle(bad_run)

    mismatched_draft = _write_fixture(tmp_path / "draft-identity")
    (mismatched_draft / "paper-draft" / "paper_draft_manifest.json").write_text(
        json.dumps({"document_status": "paper-draft", "study_id": "other-study"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="same study_id"):
        create_research_bundle(mismatched_draft)

    missing_manifest = tmp_path / "missing-manifest.zip"
    with zipfile.ZipFile(missing_manifest, "w") as archive:
        archive.writestr("study-paper-draft/file.txt", "safe")
    with pytest.raises(ValueError, match="missing bundle_manifest"):
        verify_research_bundle(missing_manifest)

    wrong_comment = tmp_path / "wrong-comment.zip"
    shutil.copyfile(directory_bundle, wrong_comment)
    with zipfile.ZipFile(wrong_comment, "a") as archive:
        archive.comment = b"bundle-manifest-sha256:" + b"0" * 64
    with pytest.raises(ValueError, match="manifest hash does not match"):
        verify_research_bundle(wrong_comment)

    type_mismatch = tmp_path / "type-mismatch.zip"
    _rewrite_bundle_manifest(
        directory_bundle,
        type_mismatch,
        lambda payload: payload["files"][0].update(type="directory"),  # type: ignore[index,union-attr]
    )
    with pytest.raises(ValueError, match="type mismatch"):
        verify_research_bundle(type_mismatch)

    size_mismatch = tmp_path / "size-mismatch.zip"

    def change_size(payload: dict[str, object]) -> None:
        """Increment one valid size without changing member bytes."""
        entry = payload["files"][0]  # type: ignore[index]
        entry["size_bytes"] += 1  # type: ignore[index,operator]

    _rewrite_bundle_manifest(directory_bundle, size_mismatch, change_size)
    with pytest.raises(ValueError, match="size mismatch"):
        verify_research_bundle(size_mismatch)
