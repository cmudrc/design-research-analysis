from __future__ import annotations

import importlib
import importlib.metadata as metadata

import pytest

import design_research_analysis._version as version_module


def test_version_module_falls_back_when_distribution_metadata_is_missing() -> None:
    def _missing_version(_: str) -> str:
        raise metadata.PackageNotFoundError

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(metadata, "version", _missing_version)
        reloaded = importlib.reload(version_module)
        assert reloaded.__version__ == "0+unknown"

    restored = importlib.reload(version_module)
    assert restored.__version__ != "0+unknown"
