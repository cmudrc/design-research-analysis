from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import design_research_analysis.runtime as runtime_module
from design_research_analysis.runtime import capture_run_context, write_run_manifest


def test_is_notebook_import_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module, "is_google_colab", lambda: False)
    monkeypatch.setattr(
        runtime_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("missing")),
    )
    assert runtime_module.is_notebook() is False

    monkeypatch.setattr(
        runtime_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(),
    )
    assert runtime_module.is_notebook() is False

    monkeypatch.setattr(
        runtime_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(get_ipython=lambda: None),
    )
    assert runtime_module.is_notebook() is False


def test_get_git_context_failure_paths_and_success(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []

    responses = iter(
        [
            (True, "/repo"),
            (False, ""),
        ]
    )
    monkeypatch.setattr(runtime_module, "_run_git_command", lambda args: next(responses))
    context = runtime_module._get_git_context(warnings)
    assert context["commit"] is None
    assert "failed to resolve current commit" in warnings[0]

    warnings = []
    responses = iter(
        [
            (True, "/repo"),
            (True, "abc123"),
            (False, ""),
        ]
    )
    monkeypatch.setattr(runtime_module, "_run_git_command", lambda args: next(responses))
    context = runtime_module._get_git_context(warnings)
    assert context["branch"] is None
    assert "failed to resolve current branch" in warnings[0]

    warnings = []
    responses = iter(
        [
            (True, "/repo"),
            (True, "abc123"),
            (True, ""),
            (False, ""),
        ]
    )
    monkeypatch.setattr(runtime_module, "_run_git_command", lambda args: next(responses))
    context = runtime_module._get_git_context(warnings)
    assert context["repo_root"] is None
    assert "failed to inspect working tree status" in warnings[0]

    warnings = []
    responses = iter(
        [
            (True, "/repo"),
            (True, "abc123"),
            (True, ""),
            (True, " M tracked.py"),
        ]
    )
    monkeypatch.setattr(runtime_module, "_run_git_command", lambda args: next(responses))
    context = runtime_module._get_git_context(warnings)
    assert context == {
        "commit": "abc123",
        "branch": None,
        "is_dirty": True,
        "repo_root": "/repo",
    }
    assert warnings == []


def test_get_package_versions_uses_metadata_and_module_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module, "_TRACKED_PACKAGES", ("pkg-a", "pkg-b", "pkg-c"))

    def _fake_version(package_name: str) -> str:
        if package_name == "pkg-a":
            return "1.2.3"
        raise runtime_module.metadata.PackageNotFoundError

    def _fake_import(name: str) -> object:
        if name == "pkg_b":
            return SimpleNamespace(__version__="4.5.6")
        if name == "pkg_c":
            return SimpleNamespace(__version__=7)
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime_module.metadata, "version", _fake_version)
    monkeypatch.setattr(runtime_module.importlib, "import_module", _fake_import)

    versions = runtime_module._get_package_versions()

    assert versions == {"pkg-a": "1.2.3", "pkg-b": "4.5.6"}


def test_capture_run_context_missing_input_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        capture_run_context(input_paths=[tmp_path / "missing.txt"])


def test_write_run_manifest_wraps_directory_and_write_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mkdir_target = tmp_path / "nested" / "manifest.json"
    monkeypatch.setattr(
        Path, "mkdir", lambda self, **kwargs: (_ for _ in ()).throw(OSError("nope"))
    )
    with pytest.raises(ValueError, match="Failed to create output directory"):
        write_run_manifest({}, mkdir_target)

    monkeypatch.undo()

    def _fake_open(self: Path, *args: object, **kwargs: object) -> object:
        raise OSError("blocked")

    monkeypatch.setattr(Path, "open", _fake_open)
    with pytest.raises(ValueError, match="Failed to write run manifest"):
        write_run_manifest({}, tmp_path / "manifest.json")
