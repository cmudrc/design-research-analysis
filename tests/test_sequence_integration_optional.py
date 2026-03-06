from __future__ import annotations

import importlib.util
import os

import pytest

from design_research_analysis.sequence import decode_hmm, fit_text_gaussian_hmm


def _integration_enabled() -> bool:
    return os.environ.get("DESIGN_RESEARCH_ANALYSIS_RUN_EMBEDDING_INTEGRATION") == "1"


def _deps_available() -> bool:
    return (
        importlib.util.find_spec("hmmlearn") is not None
        and importlib.util.find_spec("sentence_transformers") is not None
    )


@pytest.mark.skipif(not _integration_enabled(), reason="Integration test disabled by default")
@pytest.mark.skipif(not _deps_available(), reason="Required optional dependencies are unavailable")
def test_fit_text_gaussian_hmm_with_real_embeddings() -> None:
    texts = [
        "robot picks up the sample",
        "robot places the sample",
        "camera tracks the robot arm",
        "camera briefly loses lock",
        "operator resets the scene",
    ]

    result = fit_text_gaussian_hmm(
        texts,
        n_states=2,
        model_name="all-MiniLM-L6-v2",
        seed=0,
        n_iter=20,
    )
    decoded = decode_hmm(result, result.means)

    assert result.means.shape[0] == 2
    assert decoded.states.shape[0] == 2
