"""Runtime environment and reproducibility helpers."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform as _platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from sys import modules as _modules
from typing import Any

_TRACKED_PACKAGES = (
    "design-research-analysis",
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "statsmodels",
)


def is_google_colab() -> bool:
    """Return ``True`` when running inside Google Colab."""
    return "google.colab" in _modules


def is_notebook() -> bool:
    """Return ``True`` when running in a notebook-style interactive shell."""
    if is_google_colab():
        return True

    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    if shell is None:
        return False
    return type(shell).__name__ == "ZMQInteractiveShell"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _run_git_command(args: list[str]) -> tuple[bool, str]:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.returncode == 0, result.stdout.strip()


def _get_git_context(warnings: list[str]) -> dict[str, Any]:
    git_context: dict[str, Any] = {
        "commit": None,
        "branch": None,
        "is_dirty": None,
        "repo_root": None,
    }

    ok, repo_root = _run_git_command(["rev-parse", "--show-toplevel"])
    if not ok:
        warnings.append("Git metadata unavailable; current working directory is not a git repo.")
        return git_context

    ok, commit = _run_git_command(["rev-parse", "HEAD"])
    if not ok:
        warnings.append("Git metadata unavailable; failed to resolve current commit.")
        return git_context

    ok, branch = _run_git_command(["branch", "--show-current"])
    if not ok:
        warnings.append("Git metadata unavailable; failed to resolve current branch.")
        return git_context

    ok, status = _run_git_command(["status", "--porcelain", "--untracked-files=no"])
    if not ok:
        warnings.append("Git metadata unavailable; failed to inspect working tree status.")
        return git_context

    git_context["repo_root"] = repo_root
    git_context["commit"] = commit
    git_context["branch"] = branch or None
    git_context["is_dirty"] = bool(status)
    return git_context


def _get_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in _TRACKED_PACKAGES:
        version_value: str | None = None
        try:
            version_value = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            import_name = package_name.replace("-", "_")
            try:
                module = importlib.import_module(import_name)
            except Exception:
                version_value = None
            else:
                module_version = getattr(module, "__version__", None)
                if isinstance(module_version, str):
                    version_value = module_version
        if version_value is not None:
            versions[package_name] = version_value
    return versions


def capture_run_context(
    *,
    seed: int | None = None,
    input_paths: list[str | Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture deterministic provenance metadata for an analysis run."""
    warnings: list[str] = []
    resolved_inputs: list[dict[str, Any]] = []

    for raw_path in input_paths or []:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        resolved_inputs.append(
            {
                "path": str(path),
                "sha256": _hash_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git": _get_git_context(warnings),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "system": _platform.system(),
            "release": _platform.release(),
            "machine": _platform.machine(),
            "node": _platform.node(),
        },
        "packages": _get_package_versions(),
        "random_seed": seed,
        "inputs": resolved_inputs,
        "extra": dict(extra or {}),
        "warnings": warnings,
    }


def write_run_manifest(context: dict[str, Any], outpath: str | Path) -> Path:
    """Write a run-context dictionary to a JSON manifest file."""
    output_path = Path(outpath)
    if output_path.suffix.lower() != ".json":
        raise ValueError("Run manifests must be written to a .json file.")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"Failed to create output directory '{output_path.parent}': {exc}"
        ) from exc

    try:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(context, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        raise ValueError(f"Failed to write run manifest '{output_path}': {exc}") from exc
    return output_path.resolve()


def attach_provenance(result: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``result`` enriched with a ``provenance`` field."""
    enriched = dict(result)
    enriched["provenance"] = context
    return enriched


__all__ = [
    "attach_provenance",
    "capture_run_context",
    "is_google_colab",
    "is_notebook",
    "write_run_manifest",
]
