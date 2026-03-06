from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

import design_research_analysis.runtime as runtime_module
from design_research_analysis.runtime import (
    attach_provenance,
    capture_run_context,
    is_google_colab,
    is_notebook,
    write_run_manifest,
)


def test_is_google_colab_true_and_false(monkeypatch) -> None:
    monkeypatch.setitem(runtime_module._modules, "google.colab", object())
    assert is_google_colab() is True

    monkeypatch.delitem(runtime_module._modules, "google.colab", raising=False)
    assert is_google_colab() is False


def test_is_notebook_paths(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "is_google_colab", lambda: True)
    assert is_notebook() is True

    monkeypatch.setattr(runtime_module, "is_google_colab", lambda: False)

    class ZMQInteractiveShell:
        pass

    class TerminalInteractiveShell:
        pass

    monkeypatch.setitem(
        sys.modules,
        "IPython",
        SimpleNamespace(get_ipython=lambda: ZMQInteractiveShell()),
    )
    assert is_notebook() is True

    monkeypatch.setitem(
        sys.modules,
        "IPython",
        SimpleNamespace(get_ipython=lambda: TerminalInteractiveShell()),
    )
    assert is_notebook() is False


def test_capture_run_context_hashes_known_file(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    payload = b"research-data\n"
    path.write_bytes(payload)

    context = capture_run_context(seed=11, input_paths=[path])

    assert set(context.keys()) == {
        "timestamp_utc",
        "git",
        "python",
        "platform",
        "packages",
        "random_seed",
        "inputs",
        "extra",
        "warnings",
    }
    assert context["random_seed"] == 11
    assert context["inputs"][0]["path"] == str(path.resolve())
    assert context["inputs"][0]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_capture_run_context_handles_non_git_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    context = capture_run_context()

    assert context["git"] == {
        "commit": None,
        "branch": None,
        "is_dirty": None,
        "repo_root": None,
    }
    assert context["warnings"]


def test_write_manifest_rejects_non_json(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_run_manifest({}, tmp_path / "manifest.txt")


def test_write_manifest_and_attach_provenance(tmp_path) -> None:
    context = {"seed": 3}
    manifest_path = write_run_manifest(context, tmp_path / "manifest.json")

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == context

    result = {"estimate": 1.23}
    attached = attach_provenance(result, context)
    assert result == {"estimate": 1.23}
    assert attached["provenance"] == context
