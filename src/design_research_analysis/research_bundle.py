"""Create and verify narrow, inspection-oriented paper-draft bundles."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

from .paper import load_analysis_result

BUNDLE_SCHEMA_VERSION = "0.1.0"
"""Version of the verified paper-bundle manifest."""

_BUNDLE_ROOT = "study-paper-draft"
_MANIFEST_NAME = "bundle_manifest.json"
_COMMENT_PREFIX = "bundle-manifest-sha256:"
_SUPPORTED_ARTIFACT_SCHEMAS = {"0.1.0", "0.2.0"}
_CANONICAL_FILES = (
    "study.yaml",
    "manifest.json",
    "conditions.csv",
    "runs.csv",
    "events.csv",
    "evaluations.csv",
    "hypotheses.json",
    "analysis_plan.json",
)
_PAPER_DRAFT_FILES = (
    "main.tex",
    "paper_draft.md",
    "references.bib",
    "paper_draft_manifest.json",
    "README.md",
    "sections/introduction.tex",
    "sections/background.tex",
    "sections/methods.tex",
    "sections/results.tex",
    "sections/discussion.tex",
)
_PACKAGE_ALLOWLIST = (
    "design-research",
    "design-research-problems",
    "design-research-agents",
    "design-research-experiments",
    "design-research-analysis",
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "statsmodels",
)


def create_research_bundle(
    study_output: str | Path,
    *,
    output_path: str | Path | None = None,
    supporting_data: Sequence[str | Path] = (),
    overwrite: bool = False,
) -> Path:
    """Create a deterministic ZIP for integrity checks and manual reanalysis.

    Only the canonical study files, run records, analysis records, analysis
    tables/figures, complete paper-draft tree, sanitized environment summary,
    and explicitly selected supporting data are included. Run attachments and
    unrelated files are excluded unless named through ``supporting_data``.

    Args:
        study_output: Study artifact directory or its ``manifest.json``.
        output_path: Destination ZIP. Defaults to ``study-paper-draft.zip`` in
            the study directory.
        supporting_data: Explicit relative files or directories to include.
        overwrite: Whether an existing destination may be replaced.

    Returns:
        Absolute path to the verified ZIP bundle.

    Raises:
        FileExistsError: If the destination exists without ``overwrite``.
        ValueError: If required evidence is missing or a path is unsafe.
    """
    root = _resolve_study_root(study_output)
    destination = Path(output_path) if output_path is not None else root / "study-paper-draft.zip"
    destination = destination.expanduser().absolute()
    if destination.suffix.lower() != ".zip":
        raise ValueError("Research bundles must use a .zip output path.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Research bundle already exists: {destination}.")
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError("Research bundle output must be a regular, non-symlink file path.")
    _reject_selected_destination(destination, root=root, supporting_data=supporting_data)

    payloads, study_manifest = _collect_payloads(
        root,
        supporting_data=supporting_data,
    )
    environment = _json_bytes(_sanitized_environment())
    payloads["bundle/environment.json"] = (environment, "environment", "file")
    inventory = _build_inventory(payloads)
    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_type": "verified-paper-draft",
        "study_id": str(study_manifest["study_id"]),
        "source_artifact_schema_version": str(study_manifest["schema_version"]),
        "archive_root": _BUNDLE_ROOT,
        "inventory_hash_scope": "all members except bundle_manifest.json",
        "bundle_manifest_integrity": "SHA-256 stored in the ZIP comment",
        "selected_supporting_data": [
            _resolve_supporting_path(root, path).relative_to(root).as_posix()
            for path in supporting_data
        ],
        "default_exclusions": [
            "per-run attachments",
            "unselected supporting data",
            "environment variables and host identity",
        ],
        "capabilities": [
            "integrity verification",
            "collaborator inspection",
            "manual reanalysis from retained observations",
            "continued paper-draft editing",
        ],
        "non_goals": [
            "automatic analysis replay",
            "environment recreation",
            "byte-identical model reruns",
            "live-model observation regeneration",
            "arbitrary code execution",
            "storage or tracking platform",
        ],
        "files": inventory,
    }
    manifest_bytes = _json_bytes(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_zip_path(destination)
    try:
        _write_zip(
            temporary,
            payloads=payloads,
            manifest_bytes=manifest_bytes,
        )
        verify_research_bundle(temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def verify_research_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Verify archive safety, manifest integrity, inventory, sizes, and hashes.

    Args:
        bundle_path: ZIP bundle created by :func:`create_research_bundle`.

    Returns:
        Machine-readable verification summary.

    Raises:
        ValueError: If the archive is unsafe, malformed, incomplete, or altered.
    """
    path = Path(bundle_path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise ValueError(f"Expected an existing .zip research bundle: {path}.")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("Research bundle contains duplicate archive members.")
            for info in infos:
                _validate_archive_member(info)
            manifest_name = f"{_BUNDLE_ROOT}/{_MANIFEST_NAME}"
            if manifest_name not in names:
                raise ValueError("Research bundle is missing bundle_manifest.json.")
            manifest_bytes = archive.read(manifest_name)
            expected_manifest_hash = _comment_manifest_hash(archive.comment)
            actual_manifest_hash = _sha256(manifest_bytes)
            if actual_manifest_hash != expected_manifest_hash:
                raise ValueError("Research bundle manifest hash does not match the ZIP comment.")
            manifest = _load_bundle_manifest(manifest_bytes)
            inventory = _inventory_by_path(manifest)
            expected_names = {f"{_BUNDLE_ROOT}/{relative_path}" for relative_path in inventory} | {
                manifest_name
            }
            if set(names) != expected_names:
                raise ValueError("Research bundle members do not match the manifest inventory.")
            info_by_name = {info.filename: info for info in infos}
            for relative_path, entry in inventory.items():
                member = f"{_BUNDLE_ROOT}/{relative_path}"
                declared_type = entry.get("type")
                if declared_type not in {"file", "directory"}:
                    raise ValueError(f"Research bundle has an invalid type for {relative_path!r}.")
                if info_by_name[member].is_dir() != (declared_type == "directory"):
                    raise ValueError(f"Research bundle type mismatch for {relative_path!r}.")
                content = archive.read(member)
                if len(content) != entry["size_bytes"]:
                    raise ValueError(f"Research bundle size mismatch for {relative_path!r}.")
                if _sha256(content) != entry["sha256"]:
                    raise ValueError(f"Research bundle hash mismatch for {relative_path!r}.")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid research bundle {path}: {exc}") from exc
    return {
        "valid": True,
        "bundle_schema_version": manifest["bundle_schema_version"],
        "study_id": manifest["study_id"],
        "file_count": len(inventory) + 1,
        "bundle_manifest_sha256": actual_manifest_hash,
    }


def _resolve_study_root(study_output: str | Path) -> Path:
    """Resolve a study directory from the directory or its manifest path."""
    candidate = Path(study_output).expanduser()
    if candidate.is_file() and candidate.name == "manifest.json":
        candidate = candidate.parent
    root = candidate.resolve()
    if not root.is_dir():
        raise ValueError("Expected a study output directory or its manifest.json.")
    return root


def _reject_selected_destination(
    destination: Path,
    *,
    root: Path,
    supporting_data: Sequence[str | Path],
) -> None:
    """Prevent output recursion through a selected source tree."""
    selected_roots = [
        root / "paper-draft",
        root / "artifacts" / "runs",
        root / "artifacts" / "analysis",
    ]
    selected_roots.extend(_resolve_supporting_path(root, path) for path in supporting_data)
    for selected in selected_roots:
        if selected.is_dir() and (destination == selected or selected in destination.parents):
            raise ValueError("Research bundle output cannot be inside a selected source directory.")


def _collect_payloads(
    root: Path,
    *,
    supporting_data: Sequence[str | Path],
) -> tuple[dict[str, tuple[bytes, str, str]], Mapping[str, Any]]:
    """Collect and validate every selected source member."""
    payloads: dict[str, tuple[bytes, str, str]] = {}
    _require_files(root, _CANONICAL_FILES)
    for relative in _CANONICAL_FILES:
        _add_file(payloads, root, root / relative, category="canonical-study")
    manifest = _load_json_object(root / "manifest.json", label="manifest.json")
    if (
        not str(manifest.get("study_id", "")).strip()
        or not str(manifest.get("schema_version", "")).strip()
    ):
        raise ValueError("manifest.json must declare study_id and schema_version.")
    if manifest["schema_version"] not in _SUPPORTED_ARTIFACT_SCHEMAS:
        supported = ", ".join(sorted(_SUPPORTED_ARTIFACT_SCHEMAS))
        raise ValueError(
            f"Unsupported experiment artifact schema {manifest['schema_version']!r}; "
            f"expected one of: {supported}."
        )

    for optional in ("component_metadata.json", "analysis_results.json"):
        if (root / optional).is_file():
            _add_file(payloads, root, root / optional, category="component-metadata")

    run_records = sorted((root / "artifacts" / "runs").glob("*/run.json"))
    if not run_records:
        raise ValueError("A verified paper bundle requires at least one per-run run.json record.")
    for run_record in run_records:
        run_payload = _load_json_object(run_record, label=str(run_record.relative_to(root)))
        if (
            not str(run_payload.get("run_id", "")).strip()
            or not str(run_payload.get("status", "")).strip()
        ):
            raise ValueError(f"{run_record.relative_to(root)} must declare run_id and status.")
        _add_file(payloads, root, run_record, category="run-evidence")
        observations = run_record.with_name("observations.jsonl")
        if observations.is_file():
            _add_file(payloads, root, observations, category="run-evidence")

    analysis_records = sorted((root / "artifacts" / "analysis" / "results").glob("*.json"))
    if not analysis_records:
        raise ValueError("A verified paper bundle requires at least one analysis-result record.")
    for analysis_record in analysis_records:
        record = load_analysis_result(analysis_record)
        for artifact in (*record.tables, *record.figures):
            artifact_path = _resolve_relative(
                root,
                artifact,
                label="analysis artifact",
                must_exist=False,
            )
            if not artifact_path.is_file():
                raise ValueError(f"Analysis result references a missing artifact: {artifact!r}.")
            _add_file(payloads, root, artifact_path, category="analysis-artifact")
        _add_file(payloads, root, analysis_record, category="analysis-result")
    for folder in ("tables", "figures"):
        path = root / "artifacts" / "analysis" / folder
        if path.is_dir():
            _add_tree(payloads, root, path, category="analysis-artifact")

    paper_draft = root / "paper-draft"
    _require_files(paper_draft, _PAPER_DRAFT_FILES)
    draft_manifest = _load_json_object(
        paper_draft / "paper_draft_manifest.json",
        label="paper_draft_manifest.json",
    )
    if draft_manifest.get("document_status") != "paper-draft":
        raise ValueError("paper_draft_manifest.json must declare document_status 'paper-draft'.")
    if draft_manifest.get("study_id") != manifest.get("study_id"):
        raise ValueError("Paper-draft and canonical manifests must declare the same study_id.")
    _add_tree(payloads, root, paper_draft, category="paper-draft")

    for selected in supporting_data:
        path = _resolve_supporting_path(root, selected)
        if path.is_dir():
            _add_tree(payloads, root, path, category="supporting-data")
        elif path.is_file():
            _add_file(payloads, root, path, category="supporting-data")
        else:
            raise ValueError(f"Selected supporting data does not exist: {selected!s}.")
    return payloads, manifest


def _require_files(root: Path, relative_paths: Sequence[str]) -> None:
    """Require a set of files beneath one directory."""
    missing = [relative for relative in relative_paths if not (root / relative).is_file()]
    if missing:
        raise ValueError("Missing required bundle files: " + ", ".join(missing) + ".")


def _add_tree(
    payloads: dict[str, tuple[bytes, str, str]],
    root: Path,
    tree: Path,
    *,
    category: str,
) -> None:
    """Add a selected directory tree, including empty directories."""
    _validate_source_path(root, tree)
    for path in (tree, *sorted(tree.rglob("*"))):
        _validate_source_path(root, path)
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            payloads.setdefault(f"{relative}/", (b"", category, "directory"))
        elif path.is_file():
            _add_file(payloads, root, path, category=category)


def _add_file(
    payloads: dict[str, tuple[bytes, str, str]],
    root: Path,
    path: Path,
    *,
    category: str,
) -> None:
    """Add one validated source file unless a required category already selected it."""
    _validate_source_path(root, path)
    relative = path.relative_to(root).as_posix()
    payloads.setdefault(relative, (path.read_bytes(), category, "file"))


def _validate_source_path(root: Path, path: Path) -> None:
    """Reject missing paths, symlinks, and paths outside the study root."""
    resolved_root = root.resolve()
    unresolved = path if path.is_absolute() else root / path
    try:
        relative = unresolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Bundle source escapes the study root: {path}.") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Bundle source cannot traverse a symlink: {path}.")
    resolved = unresolved.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Bundle source escapes the study root: {path}.")
    if not unresolved.exists():
        raise ValueError(f"Bundle source does not exist: {path}.")


def _resolve_supporting_path(root: Path, raw_path: str | Path) -> Path:
    """Resolve one explicitly selected supporting-data path."""
    path = _resolve_relative(root, raw_path, label="supporting-data")
    if path == root:
        raise ValueError(
            "Select specific supporting-data files or directories, not the study root."
        )
    return path


def _resolve_relative(
    root: Path,
    raw_path: str | Path,
    *,
    label: str,
    must_exist: bool = True,
) -> Path:
    """Resolve a safe relative path beneath the study root."""
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} paths must be relative to the study root: {raw_path!s}.")
    path = root / relative
    if must_exist:
        _validate_source_path(root, path)
    else:
        resolved_root = root.resolve()
        resolved = path.resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise ValueError(f"{label} path escapes the study root: {raw_path!s}.")
    return path


def _load_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    """Load a JSON object with a bundle-specific error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label}: {exc.msg}.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object.")
    return payload


def _sanitized_environment() -> dict[str, Any]:
    """Capture only portable, allowlisted environment facts."""
    packages: dict[str, str] = {}
    for package_name in _PACKAGE_ALLOWLIST:
        try:
            packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return {
        "python_version": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "excluded": ["environment variables", "hostname", "executable path"],
    }


def _build_inventory(
    payloads: Mapping[str, tuple[bytes, str, str]],
) -> list[dict[str, Any]]:
    """Build the deterministic member inventory."""
    return [
        {
            "path": path,
            "type": file_type,
            "category": category,
            "size_bytes": len(content),
            "sha256": _sha256(content),
        }
        for path, (content, category, file_type) in sorted(payloads.items())
    ]


def _temporary_zip_path(destination: Path) -> Path:
    """Reserve a same-directory temporary ZIP path for atomic replacement."""
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".zip",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(raw_path)


def _write_zip(
    path: Path,
    *,
    payloads: Mapping[str, tuple[bytes, str, str]],
    manifest_bytes: bytes,
) -> None:
    """Write members with stable order, timestamps, permissions, and compression."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, (content, _category, file_type) in sorted(payloads.items()):
            _write_member(archive, relative, content, file_type=file_type)
        _write_member(archive, _MANIFEST_NAME, manifest_bytes, file_type="file")
        archive.comment = f"{_COMMENT_PREFIX}{_sha256(manifest_bytes)}".encode()


def _write_member(
    archive: zipfile.ZipFile,
    relative: str,
    content: bytes,
    *,
    file_type: str,
) -> None:
    """Write one safe bundle-rooted ZIP member."""
    archive_name = f"{_BUNDLE_ROOT}/{relative}"
    info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o40755 if file_type == "directory" else 0o100644) << 16
    archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _validate_archive_member(info: zipfile.ZipInfo) -> None:
    """Reject traversal, absolute, malformed-root, and symlink members."""
    member = PurePosixPath(info.filename)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise ValueError(f"Research bundle contains an unsafe member: {info.filename!r}.")
    if member.parts[0] != _BUNDLE_ROOT:
        raise ValueError(f"Research bundle member is outside {_BUNDLE_ROOT!r}.")
    unix_mode = (info.external_attr >> 16) & 0o170000
    if unix_mode == 0o120000:
        raise ValueError(f"Research bundle cannot contain symlinks: {info.filename!r}.")


def _comment_manifest_hash(comment: bytes) -> str:
    """Read the manifest digest from the ZIP comment."""
    try:
        value = comment.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Research bundle ZIP comment is not valid ASCII.") from exc
    if not value.startswith(_COMMENT_PREFIX):
        raise ValueError("Research bundle ZIP comment is missing the manifest hash.")
    digest = value.removeprefix(_COMMENT_PREFIX)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Research bundle ZIP comment has an invalid manifest hash.")
    return digest


def _load_bundle_manifest(content: bytes) -> Mapping[str, Any]:
    """Parse and validate the bundle manifest header."""
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Research bundle manifest is not valid UTF-8 JSON.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Research bundle manifest must be a JSON object.")
    if payload.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("Research bundle uses an unsupported bundle schema version.")
    if payload.get("archive_root") != _BUNDLE_ROOT:
        raise ValueError("Research bundle manifest declares an unexpected archive root.")
    if not str(payload.get("study_id", "")).strip():
        raise ValueError("Research bundle manifest is missing study_id.")
    return payload


def _inventory_by_path(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Validate and index inventory rows by safe relative path."""
    entries = manifest.get("files")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("Research bundle manifest files must be an array.")
    inventory: dict[str, Mapping[str, Any]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("Research bundle inventory entries must be JSON objects.")
        relative = str(raw_entry.get("path", ""))
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or not relative:
            raise ValueError(f"Research bundle manifest contains an unsafe path: {relative!r}.")
        if relative in inventory:
            raise ValueError(f"Research bundle manifest repeats path {relative!r}.")
        digest = raw_entry.get("sha256")
        size = raw_entry.get("size_bytes")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Research bundle manifest has an invalid hash for {relative!r}.")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"Research bundle manifest has an invalid size for {relative!r}.")
        inventory[relative] = raw_entry
    return inventory


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize deterministic, human-readable JSON bytes."""
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _sha256(content: bytes) -> str:
    """Return the lowercase SHA-256 digest for bytes."""
    return hashlib.sha256(content).hexdigest()
