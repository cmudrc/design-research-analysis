from __future__ import annotations

import builtins
import json
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

import design_research_analysis.sequence._backend as backend_module
import design_research_analysis.sequence.models as models_module
from design_research_analysis.sequence import decode_hmm, fit_discrete_hmm, fit_gaussian_hmm, fit_text_gaussian_hmm
from design_research_analysis.sequence._backend import HmmlearnBackend, _BaseHMMBackend, get_hmm_backend
from design_research_analysis.sequence.visualization import plot_state_graph, plot_transition_matrix


class _FakeGaussianModel:
    def __init__(self) -> None:
        self.startprob_ = np.array([0.6, 0.4])
        self.transmat_ = np.array([[0.8, 0.2], [0.1, 0.9]])
        self.means_ = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.covars_ = np.array([np.eye(2), np.eye(2)])

    def fit(self, x: np.ndarray, lengths: list[int] | None = None) -> None:
        self._fit_shape = x.shape
        self._fit_lengths = lengths

    def score(self, x: np.ndarray, lengths: list[int] | None = None) -> float:
        return float(x.shape[0])

    def decode(
        self,
        x: np.ndarray,
        lengths: list[int] | None = None,
        algorithm: str = "viterbi",
    ) -> tuple[float, np.ndarray]:
        return -1.0, np.zeros(x.shape[0], dtype=int)


class _FakeDiscreteModel:
    def __init__(self, n_symbols: int) -> None:
        self.startprob_ = np.array([0.5, 0.5])
        self.transmat_ = np.array([[0.7, 0.3], [0.2, 0.8]])
        self.emissionprob_ = np.full((2, n_symbols), 1.0 / n_symbols)
        self.n_features = n_symbols

    def fit(self, x: np.ndarray, lengths: list[int] | None = None) -> None:
        self._fit_shape = x.shape
        self._fit_lengths = lengths

    def score(self, x: np.ndarray, lengths: list[int] | None = None) -> float:
        return float(-x.shape[0])

    def decode(
        self,
        x: np.ndarray,
        lengths: list[int] | None = None,
        algorithm: str = "viterbi",
    ) -> tuple[float, np.ndarray]:
        return -2.0, np.arange(x.shape[0], dtype=int) % 2


class _FakeBackend:
    name = "hmmlearn"

    def create_gaussian_hmm(
        self,
        *,
        n_states: int,
        covariance_type: str,
        n_iter: int,
        seed: int,
    ) -> _FakeGaussianModel:
        assert n_states == 2
        assert covariance_type == "diag"
        assert n_iter == 20
        assert seed == 7
        return _FakeGaussianModel()

    def create_discrete_hmm(
        self,
        *,
        n_states: int,
        n_iter: int,
        seed: int,
        n_symbols: int,
    ) -> _FakeDiscreteModel:
        assert n_states == 2
        assert n_iter == 20
        assert seed == 7
        return _FakeDiscreteModel(n_symbols)


class _FakeNxGraph:
    def __init__(self) -> None:
        self._nodes: dict[int, dict[str, object]] = {}
        self._edges: dict[tuple[int, int], dict[str, float]] = {}

    def add_node(self, idx: int, label: str) -> None:
        self._nodes[idx] = {"label": label}

    def add_edge(self, src: int, dst: int, weight: float) -> None:
        self._edges[(src, dst)] = {"weight": weight}

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    @property
    def edges(self) -> list[tuple[int, int]]:
        return list(self._edges.keys())

    def __getitem__(self, src: int) -> dict[int, dict[str, float]]:
        return {dst: attrs for (u, dst), attrs in self._edges.items() if u == src}


def _fake_networkx_module() -> object:
    def _spring_layout(graph: _FakeNxGraph, seed: int = 0) -> dict[int, tuple[float, float]]:
        return {idx: (float(idx), float(seed + idx)) for idx in range(graph.number_of_nodes())}

    return SimpleNamespace(
        DiGraph=_FakeNxGraph,
        spring_layout=_spring_layout,
        draw_networkx_nodes=lambda *args, **kwargs: None,
        draw_networkx_labels=lambda *args, **kwargs: None,
        draw_networkx_edges=lambda *args, **kwargs: None,
        draw_networkx_edge_labels=lambda *args, **kwargs: None,
    )


def test_base_backend_methods_raise_not_implemented() -> None:
    backend = _BaseHMMBackend()
    with pytest.raises(NotImplementedError):
        backend.create_gaussian_hmm(n_states=2, covariance_type="diag", n_iter=10, seed=0)
    with pytest.raises(NotImplementedError):
        backend.create_discrete_hmm(n_states=2, n_iter=10, seed=0, n_symbols=3)


def test_hmmlearn_backend_import_error(monkeypatch) -> None:
    original_import = builtins.__import__

    def _mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "hmmlearn":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)
    with pytest.raises(ImportError, match="optional sequence dependencies"):
        HmmlearnBackend()._load_hmm_module()


def test_hmmlearn_backend_load_module_success(monkeypatch) -> None:
    original_import = builtins.__import__
    fake_hmm = SimpleNamespace(marker="ok")

    def _mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "hmmlearn":
            return SimpleNamespace(hmm=fake_hmm)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)
    assert HmmlearnBackend()._load_hmm_module() is fake_hmm


def test_hmmlearn_backend_create_methods_with_fake_hmm(monkeypatch) -> None:
    class _FakeGaussianCtor:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _FakeCategoricalCtor:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.n_features = 0

    fake_hmm_module = SimpleNamespace(
        GaussianHMM=_FakeGaussianCtor,
        CategoricalHMM=_FakeCategoricalCtor,
    )
    monkeypatch.setattr(HmmlearnBackend, "_load_hmm_module", lambda self: fake_hmm_module)

    backend = HmmlearnBackend()
    gaussian = backend.create_gaussian_hmm(n_states=3, covariance_type="full", n_iter=30, seed=11)
    discrete = backend.create_discrete_hmm(n_states=2, n_iter=20, seed=9, n_symbols=5)

    assert gaussian.kwargs["n_components"] == 3
    assert discrete.n_features == 5


def test_hmmlearn_backend_discrete_requires_categorical(monkeypatch) -> None:
    fake_hmm_module = SimpleNamespace(GaussianHMM=object)
    monkeypatch.setattr(HmmlearnBackend, "_load_hmm_module", lambda self: fake_hmm_module)

    with pytest.raises(ImportError, match="CategoricalHMM"):
        HmmlearnBackend().create_discrete_hmm(n_states=2, n_iter=10, seed=0, n_symbols=3)


def test_get_hmm_backend_supported_and_invalid() -> None:
    assert isinstance(get_hmm_backend("hmmlearn"), backend_module.HmmlearnBackend)
    with pytest.raises(ValueError, match="Unsupported HMM backend"):
        get_hmm_backend("unknown")


def test_fit_hmms_with_mock_backend(monkeypatch) -> None:
    monkeypatch.setattr(models_module, "get_hmm_backend", lambda name: _FakeBackend())

    gaussian = fit_gaussian_hmm([[1.0, 2.0], [3.0, 4.0]], n_states=2, n_iter=20, seed=7)
    assert gaussian.train_log_likelihood == 2.0
    assert gaussian.to_dict()["n_states"] == 2

    discrete = fit_discrete_hmm([["A", "B", "A"], ["B", "A"]], n_states=2, n_iter=20, seed=7)
    payload = json.dumps(discrete.to_dict())
    assert "emissionprob" in payload

    decoded_discrete = decode_hmm(discrete, [["A", "B"], ["A"]])
    assert decoded_discrete.states.shape == (3,)
    assert decoded_discrete.to_dict()["algorithm"] == "viterbi"

    decoded_gaussian = decode_hmm(gaussian, np.ones((3, 2)))
    assert decoded_gaussian.states.shape == (3,)


def test_fit_text_gaussian_hmm_default_embedder_branch(monkeypatch) -> None:
    monkeypatch.setattr(models_module, "get_hmm_backend", lambda name: _FakeBackend())
    monkeypatch.setattr(
        models_module,
        "embed_text",
        lambda texts, **kwargs: np.asarray(
            [[float(idx), float(len(text))] for idx, text in enumerate(texts)]
        ),
    )

    result = fit_text_gaussian_hmm(["alpha", "beta", "gamma"], n_states=2, n_iter=20, seed=7)
    assert result.config["embedding_provider"] == "sentence-transformers"
    assert result.config["embedding_model"] == "all-MiniLM-L6-v2"


def test_decode_hmm_validation_paths(monkeypatch) -> None:
    monkeypatch.setattr(models_module, "get_hmm_backend", lambda name: _FakeBackend())
    discrete = fit_discrete_hmm([["x", "y", "x"]], n_states=2, n_iter=20, seed=7)

    with pytest.raises(ValueError, match="viterbi, map"):
        decode_hmm(discrete, ["x", "y"], algorithm="bad")

    with pytest.raises(ValueError, match="not in the fitted vocabulary"):
        decode_hmm(discrete, ["x", "z"])

    with pytest.raises(TypeError, match="GaussianHMMResult or DiscreteHMMResult"):
        decode_hmm(object(), ["x", "y"])  # type: ignore[arg-type]


def test_private_model_helpers_and_labels(monkeypatch) -> None:
    class _CustomToken:
        def __repr__(self) -> str:
            return "custom-token"

    assert models_module._serialize_token(_CustomToken()) == "custom-token"

    with pytest.raises(ValueError, match="must not be empty"):
        models_module._normalize_token_sequences([])

    with pytest.raises(ValueError, match="either tokens or token sequences"):
        models_module._normalize_token_sequences([["a"], 3])

    with pytest.raises(ValueError, match="must contain at least one token"):
        models_module._normalize_token_sequences([[]])

    flat, lengths = models_module._flatten_token_sequences([["a", "b"], ["c"]])
    assert flat == ["a", "b", "c"]
    assert lengths == [2, 1]

    with pytest.raises(ValueError, match="texts must not be empty"):
        models_module._normalize_text_observations([])

    with pytest.raises(ValueError, match="sequence of strings"):
        models_module._normalize_text_observations([1])

    with pytest.raises(ValueError, match="Nested text sequences must not be empty"):
        models_module._normalize_text_observations([[]])

    texts, text_lengths = models_module._normalize_text_observations([["x"], ["y", "z"]])
    assert texts == ["x", "y", "z"]
    assert text_lengths == [1, 2]

    with pytest.raises(TypeError, match="Expected MarkovChainResult"):
        models_module._transition_like_matrix(object())

    with pytest.raises(TypeError, match="Expected MarkovChainResult"):
        models_module._state_labels(object())

    monkeypatch.setattr(models_module, "get_hmm_backend", lambda name: _FakeBackend())
    discrete = fit_discrete_hmm([["x", "y", "x"]], n_states=2, n_iter=20, seed=7)
    assert models_module._state_labels(discrete) == ["S0", "S1"]

    markov = models_module.fit_markov_chain([["x", "y", "z"], ["x", "z", "y"]], order=2)
    labels = models_module._state_labels(markov)
    assert all("|" in label for label in labels)


def test_model_input_validation_helpers() -> None:
    with pytest.raises(ValueError, match="2D"):
        models_module._as_float_matrix(np.ones((2, 2, 2)), name="X")

    with pytest.raises(ValueError, match="at least one"):
        models_module._as_float_matrix(np.asarray([]), name="X")

    with pytest.raises(ValueError, match="must not be empty"):
        models_module._validate_lengths([], 3)

    with pytest.raises(ValueError, match="must be positive"):
        models_module._validate_lengths([1, 0], 1)

    with pytest.raises(ValueError, match="must equal"):
        models_module._validate_lengths([1, 1], 3)


def test_fit_markov_chain_additional_branches() -> None:
    with pytest.raises(ValueError, match="order must be positive"):
        models_module.fit_markov_chain(["a", "b"], order=0)

    with pytest.raises(ValueError, match="smoothing must be non-negative"):
        models_module.fit_markov_chain(["a", "b"], smoothing=-0.5)

    result = models_module.fit_markov_chain(
        [["x"], ["x", "y", "z"]],
        order=2,
        smoothing=0.0,
    )
    assert np.allclose(result.transition_matrix.sum(axis=1), 1.0)


def test_fit_gaussian_and_discrete_validation_branches(monkeypatch) -> None:
    with pytest.raises(ValueError, match="n_states must be positive"):
        fit_gaussian_hmm(np.ones((3, 2)), n_states=0)

    with pytest.raises(ValueError, match="n_iter must be positive"):
        fit_gaussian_hmm(np.ones((3, 2)), n_iter=0)

    monkeypatch.setattr(models_module, "get_hmm_backend", lambda name: _FakeBackend())

    with pytest.raises(ValueError, match="n_states must be positive"):
        fit_discrete_hmm([["a", "b"]], n_states=0)

    with pytest.raises(ValueError, match="n_iter must be positive"):
        fit_discrete_hmm([["a", "b"]], n_iter=0)

    monkeypatch.setattr(models_module, "_flatten_token_sequences", lambda _: ([], []))
    with pytest.raises(ValueError, match="at least one token"):
        fit_discrete_hmm([["a"]], n_states=2, n_iter=20, seed=7)


def test_fit_text_gaussian_hmm_embedder_callable_branches(monkeypatch) -> None:
    monkeypatch.setattr(models_module, "get_hmm_backend", lambda name: _FakeBackend())

    result = fit_text_gaussian_hmm(
        ["alpha", "beta"],
        n_states=2,
        n_iter=20,
        seed=7,
        embedder=lambda texts: np.asarray([float(len(item)) for item in texts]),
    )
    assert result.config["embedding_provider"] == "callable"
    assert result.model._fit_shape[1] == 1

    with pytest.raises(ValueError, match="2D embedding matrix"):
        fit_text_gaussian_hmm(
            ["alpha"],
            n_states=2,
            n_iter=20,
            seed=7,
            embedder=lambda texts: np.ones((1, 1, 1)),
        )


def test_plot_state_graph_with_fake_networkx(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "networkx", _fake_networkx_module())

    matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    fig, ax = plot_state_graph(matrix, threshold=0.1)
    assert fig is ax.figure

    fig2, ax2 = plot_transition_matrix(matrix, annotate=True)
    assert fig2 is ax2.figure


def test_visualization_validation_and_ax_branches(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "networkx", _fake_networkx_module())

    with pytest.raises(ValueError, match="must be square"):
        plot_transition_matrix(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    with pytest.raises(ValueError, match="length must match"):
        plot_transition_matrix(np.eye(2), state_labels=["only-one"])

    fig, ax = plt.subplots()
    fig_out, ax_out = plot_transition_matrix(np.eye(2), ax=ax, annotate=False)
    assert fig_out is fig
    assert ax_out is ax
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    fig2_out, ax2_out = plot_state_graph(np.eye(2), ax=ax2, threshold=0.1)
    assert fig2_out is fig2
    assert ax2_out is ax2
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    fig3_out, ax3_out = plot_state_graph(np.zeros((0, 0)), ax=ax3)
    assert fig3_out is fig3
    assert ax3_out is ax3
    plt.close(fig3)


def test_plot_state_graph_threshold_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        plot_state_graph(np.array([[1.0]]), threshold=-0.1)
