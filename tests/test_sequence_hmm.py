from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

import design_research_analysis.sequence.models as models_module
from design_research_analysis.sequence import decode_hmm, fit_discrete_hmm, fit_gaussian_hmm, fit_text_gaussian_hmm


def _hmm_available() -> bool:
    return importlib.util.find_spec("hmmlearn") is not None


@pytest.mark.skipif(not _hmm_available(), reason="hmmlearn is unavailable")
def test_fit_gaussian_hmm_and_decode_shapes() -> None:
    rng = np.random.default_rng(4)
    x1 = rng.normal(loc=-2.0, scale=0.3, size=(40, 3))
    x2 = rng.normal(loc=2.0, scale=0.3, size=(40, 3))
    x = np.vstack([x1, x2])

    result = fit_gaussian_hmm(x, lengths=[40, 40], n_states=2, n_iter=40, seed=5)
    decoded = decode_hmm(result, x, lengths=[40, 40])

    assert result.means.shape == (2, 3)
    assert result.transmat.shape == (2, 2)
    assert decoded.states.shape == (80,)
    assert np.isfinite(result.train_log_likelihood)
    assert np.isfinite(decoded.log_probability)


@pytest.mark.skipif(not _hmm_available(), reason="hmmlearn is unavailable")
def test_fit_discrete_hmm_and_decode_tokens() -> None:
    sequences = [
        ["S", "NP", "VP", "END"],
        ["S", "NP", "PP", "VP", "END"],
        ["S", "ADV", "VP", "END"],
    ]

    result = fit_discrete_hmm(sequences, n_states=3, n_iter=60, seed=2)
    decoded = decode_hmm(result, sequences)

    assert result.emissionprob.shape[1] == len(result.vocab)
    assert np.allclose(result.transmat.sum(axis=1), 1.0)
    assert decoded.states.shape[0] == sum(len(seq) for seq in sequences)


@pytest.mark.skipif(not _hmm_available(), reason="hmmlearn is unavailable")
def test_decode_discrete_hmm_rejects_unknown_token() -> None:
    result = fit_discrete_hmm([["A", "B", "A", "B"]], n_states=2, n_iter=30, seed=1)

    with pytest.raises(ValueError, match="not in the fitted vocabulary"):
        decode_hmm(result, ["A", "C", "A"])


@pytest.mark.skipif(not _hmm_available(), reason="hmmlearn is unavailable")
def test_fit_text_gaussian_hmm_with_custom_embedder() -> None:
    texts = ["robot picks object", "robot places object", "camera tracks robot"]

    def _embed(items: list[str]) -> np.ndarray:
        return np.asarray([[len(item), item.count("o")] for item in items], dtype=float)

    result = fit_text_gaussian_hmm(texts, n_states=2, embedder=_embed, n_iter=30, seed=0)

    assert result.config["embedding_provider"] == "callable"
    assert result.means.shape[1] == 2
    assert "train_log_likelihood" in json.dumps(result.to_dict())


def test_fit_gaussian_hmm_surfaces_backend_import_error(monkeypatch) -> None:
    def _raise_backend(_name: str) -> None:
        raise ImportError(
            "Sequence modeling requires optional sequence dependencies. "
            "Install with `pip install design-research-analysis[seq]`."
        )

    monkeypatch.setattr(models_module, "get_hmm_backend", _raise_backend)

    with pytest.raises(ImportError, match="optional sequence dependencies"):
        fit_gaussian_hmm(np.ones((4, 2)))
