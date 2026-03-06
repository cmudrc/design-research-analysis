from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from design_research_analysis.sequence import embed_text


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        assert batch_size == 8
        assert show_progress_bar is False
        assert normalize_embeddings is False
        assert convert_to_numpy is True
        return np.asarray([[float(len(text)), float(idx)] for idx, text in enumerate(texts)])


def test_embed_text_with_mocked_sentence_transformers(monkeypatch) -> None:
    fake_module = SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    embedded = embed_text(
        ["alpha", "beta", "gamma"],
        model_name="fake-model",
        normalize=False,
        batch_size=8,
        device="cpu",
    )

    assert embedded.shape == (3, 2)
    assert embedded[0, 0] == 5.0


def test_embed_text_surfaces_optional_dependency_error(monkeypatch) -> None:
    original_import = builtins.__import__

    def _mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers":
            raise ImportError("simulated missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)

    with pytest.raises(ImportError, match="pip install design-research-analysis\\[embeddings\\]"):
        embed_text(["a", "b"])


def test_embed_text_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must contain at least one item"):
        embed_text([])
