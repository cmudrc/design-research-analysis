from __future__ import annotations

import builtins
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pytest

import design_research_analysis.embedding_maps as maps_module
from design_research_analysis.embedding_maps import (
    EmbeddingMapResult,
    EmbeddingResult,
    _kmeans,
    build_embedding_map,
    cluster_embedding_map,
    compare_embedding_maps,
    compute_design_space_coverage,
    compute_idea_space_trajectory,
    embed_records,
    plot_embedding_map,
    plot_embedding_map_grid,
)


def test_embedding_and_map_results_support_serialization_and_comparison() -> None:
    left_embedding = EmbeddingResult(
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        record_ids=["r1", "r2"],
        texts=["alpha", "beta"],
        config={"provider": "callable"},
    )
    right_embedding = EmbeddingResult(
        embeddings=np.asarray([[1.0, 2.0], [2.5, 4.5]]),
        record_ids=["r1", "r2"],
        texts=["alpha", "beta"],
        config={"provider": "callable"},
    )

    embedding_payload = left_embedding.to_dict()
    embedding_difference = left_embedding - right_embedding

    assert embedding_payload["shape"] == [2, 2]
    assert embedding_difference.metric == "embedding_profile"
    assert embedding_difference.details["left_record_ids"] == ["r1", "r2"]

    left_projection = EmbeddingMapResult(
        coordinates=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        record_ids=["r1", "r2"],
        method="pca",
        config={"n_components": 2},
        explained_variance_ratio=[0.75, 0.25],
    )
    right_projection = EmbeddingMapResult(
        coordinates=np.asarray([[0.1, 0.1], [0.3, 0.5]]),
        record_ids=["r1", "r2"],
        method="umap",
        config={"n_components": 2},
    )

    projection_payload = left_projection.to_dict()
    projection_effect = left_projection / right_projection

    assert projection_payload["explained_variance_ratio"] == [0.75, 0.25]
    assert projection_effect.metric == "embedding_map_profile"
    assert projection_effect.details["methods"] == ["pca", "umap"]


def test_embed_records_uses_mappers_and_custom_embedder() -> None:
    rows = [
        {"timestamp": "2026-01-01T10:00:01Z", "text": "", "record_id": ""},
        {"timestamp": "2026-01-01T10:00:00Z", "text": "seed", "record_id": None},
    ]

    result = embed_records(
        rows,
        embedder=lambda texts: np.asarray([[len(text), index] for index, text in enumerate(texts)]),
        text_mapper=lambda row: f"filled:{row['timestamp']}",
        record_id_mapper=lambda row: f"id:{row['timestamp']}",
        batch_size=8,
        device="cpu",
    )

    assert result.record_ids == [
        "id:2026-01-01 10:00:00+00:00",
        "id:2026-01-01 10:00:01+00:00",
    ]
    assert result.texts == ["seed", "filled:2026-01-01 10:00:01+00:00"]
    assert result.config["provider"] == "callable"
    assert result.config["model_name"] == "custom"


def test_embed_records_validation_errors() -> None:
    with pytest.raises(ValueError, match="at least one text record"):
        embed_records([], embedder=lambda texts: np.asarray(texts, dtype=float))

    with pytest.raises(ValueError, match="missing text"):
        embed_records(
            [{"timestamp": "2026-01-01T10:00:00Z", "text": None}],
            embedder=lambda texts: np.asarray(texts, dtype=float),
        )

    with pytest.raises(ValueError, match="2D embedding matrix"):
        embed_records(
            [{"timestamp": "2026-01-01T10:00:00Z", "text": "hello"}],
            embedder=lambda texts: np.asarray([1.0, 2.0]),
        )

    with pytest.raises(ValueError, match="rows must match"):
        embed_records(
            [{"timestamp": "2026-01-01T10:00:00Z", "text": "hello"}],
            embedder=lambda texts: np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        )

    with pytest.raises(ValueError, match="record_id values must be unique"):
        embed_records(
            [
                {"timestamp": "2026-01-01T10:00:00Z", "text": "hello", "record_id": "same"},
                {"timestamp": "2026-01-01T10:00:01Z", "text": "again", "record_id": "same"},
            ],
            embedder=lambda texts: np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        )


def test_embed_records_builtin_provider_uses_embed_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_embed_text(
        texts: list[str],
        *,
        model_name: str,
        normalize: bool,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        captured["texts"] = list(texts)
        captured["model_name"] = model_name
        captured["normalize"] = normalize
        captured["batch_size"] = batch_size
        captured["device"] = device
        return np.asarray([[0.1, 0.2]])

    monkeypatch.setattr(maps_module, "embed_text", _fake_embed_text)

    result = embed_records(
        [{"timestamp": "2026-01-01T10:00:00Z", "text": "hello", "record_id": "r1"}],
        model_name="mini-model",
        normalize=False,
        batch_size=4,
        device="cpu",
    )

    assert result.config["provider"] == "sentence-transformers"
    assert captured == {
        "texts": ["hello"],
        "model_name": "mini-model",
        "normalize": False,
        "batch_size": 4,
        "device": "cpu",
    }


def test_build_embedding_map_validation_and_constant_pca_case() -> None:
    with pytest.raises(ValueError, match="2D matrix"):
        build_embedding_map(np.asarray([1.0, 2.0, 3.0]), method="pca")
    with pytest.raises(ValueError, match="at least one row"):
        build_embedding_map(np.empty((0, 2)), method="pca")
    with pytest.raises(ValueError, match="must be positive"):
        build_embedding_map(np.ones((2, 2)), method="pca", n_components=0)
    with pytest.raises(ValueError, match="Unsupported method"):
        build_embedding_map(np.ones((2, 2)), method="bogus")
    with pytest.raises(ValueError, match="record_ids length"):
        build_embedding_map(np.ones((2, 2)), method="pca", record_ids=["r1"])
    with pytest.raises(ValueError, match="record_ids must be unique"):
        build_embedding_map(np.ones((2, 2)), method="pca", record_ids=["r1", "r1"])

    constant = build_embedding_map(np.ones((3, 2)), method="pca", n_components=2)
    assert constant.explained_variance_ratio == [0.0, 0.0]

    with pytest.raises(ValueError, match="methods must contain"):
        compare_embedding_maps(np.ones((2, 2)), methods=[])
    with pytest.raises(ValueError, match="methods must be unique"):
        compare_embedding_maps(np.ones((2, 2)), methods=["pca", "PCA"])


def test_design_space_coverage_accepts_result_objects_and_degenerate_geometry() -> None:
    embedding = EmbeddingResult(
        embeddings=np.asarray([[0.0, 0.0, 0.0]]),
        record_ids=["r1"],
        texts=["one"],
    )
    singleton = compute_design_space_coverage(embedding)

    assert singleton["config"]["input_source"] == "embedding_result"
    assert singleton["pairwise_spread"]["n_pairs"] == 0
    assert singleton["warnings"]

    mapped = EmbeddingMapResult(
        coordinates=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        record_ids=["r1", "r2"],
        method="pca",
    )
    two_points = compute_design_space_coverage(mapped)
    assert two_points["config"]["input_source"] == "embedding_map_result"
    assert two_points["convex_hull"]["supported"] is False

    collinear = compute_design_space_coverage(np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    assert collinear["convex_hull"]["area"] is None
    assert any("collinear" in warning for warning in collinear["warnings"])

    with pytest.raises(ValueError, match="finite numeric values"):
        compute_design_space_coverage(np.asarray([[0.0, np.inf]]))


def test_idea_space_trajectory_singletons_and_input_validation() -> None:
    trajectory = compute_idea_space_trajectory(
        np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        timestamps=[np.int64(2), None],
        groups=["A", ""],
    )

    assert trajectory["groups"]["A"]["net_displacement"] == 0.0
    assert trajectory["groups"]["__all__"]["ordered_timestamps"] == [None]
    assert len(trajectory["warnings"]) == 2

    with pytest.raises(ValueError, match="timestamps must have the same length"):
        compute_idea_space_trajectory(np.ones((2, 2)), timestamps=[1])


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="sklearn unavailable",
)
def test_cluster_embedding_map_agglomerative_and_kmeans_validation() -> None:
    projection = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.2],
        ]
    )

    agglomerative = cluster_embedding_map(projection, method="agglomerative", n_clusters=2)

    assert agglomerative["method"] == "agglomerative"
    assert agglomerative["centers"] is None
    assert len(agglomerative["labels"]) == 4

    with pytest.raises(ValueError, match="2D matrix"):
        cluster_embedding_map(np.asarray([1.0, 2.0]), method="kmeans")
    with pytest.raises(ValueError, match="at least one row"):
        cluster_embedding_map(np.empty((0, 2)), method="kmeans")
    with pytest.raises(ValueError, match="Unsupported method"):
        cluster_embedding_map(projection, method="bogus")


def test_kmeans_validation_and_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="must be positive"):
        _kmeans(points, n_clusters=0, random_state=0, max_iter=10)
    with pytest.raises(ValueError, match="cannot exceed"):
        _kmeans(points, n_clusters=3, random_state=0, max_iter=10)

    original_import = builtins.__import__

    def _raising_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pacmap":
            raise ImportError("blocked")
        if name == "umap":
            raise ImportError("blocked")
        if name == "trimap":
            raise ImportError("blocked")
        if name == "sklearn.manifold":
            raise ImportError("blocked")
        if name == "sklearn.cluster":
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raising_import)

    with pytest.raises(ImportError, match="optional dependencies"):
        build_embedding_map(np.ones((4, 2)), method="tsne")
    with pytest.raises(ImportError, match="optional dependencies"):
        build_embedding_map(np.ones((4, 2)), method="umap")
    with pytest.raises(ImportError, match="optional dependencies"):
        build_embedding_map(np.ones((4, 2)), method="pacmap")
    with pytest.raises(ImportError, match="optional dependencies"):
        build_embedding_map(np.ones((4, 2)), method="trimap")
    with pytest.raises(ImportError, match="optional dependencies"):
        cluster_embedding_map(np.ones((4, 2)), method="kmeans")
    with pytest.raises(ImportError, match="optional dependencies"):
        cluster_embedding_map(np.ones((4, 2)), method="agglomerative")


def test_plot_embedding_map_validation_edges() -> None:
    embedding_map = EmbeddingMapResult(
        coordinates=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        record_ids=["r1", "r2"],
        method="pca",
    )
    rows = [
        {"record_id": "r1", "trace": "t", "step": "1", "value": "1.0"},
        {"record_id": "r2", "trace": "t", "step": "2", "value": "1.0"},
    ]

    fig, ax = plt.subplots()
    returned_fig, returned_ax = plot_embedding_map(embedding_map, rows, value_column="value", ax=ax)
    assert returned_fig is fig
    assert returned_ax is ax
    plt.close(fig)

    with pytest.raises(ValueError, match="plotting requires a 2D embedding map"):
        plot_embedding_map(
            EmbeddingMapResult(
                coordinates=np.ones((2, 3)),
                record_ids=["r1", "r2"],
                method="pca",
            ),
            rows,
        )
    plt.close("all")

    with pytest.raises(ValueError, match="order_column is required"):
        plot_embedding_map(embedding_map, rows, trace_column="trace")
    plt.close("all")

    with pytest.raises(ValueError, match="missing 'record_id'"):
        plot_embedding_map(embedding_map, [{"trace": "t", "step": 1, "value": 1.0}])
    plt.close("all")

    with pytest.raises(ValueError, match="Duplicate record_id"):
        plot_embedding_map(embedding_map, [rows[0], rows[0]])
    plt.close("all")

    with pytest.raises(ValueError, match="missing map record IDs"):
        plot_embedding_map(embedding_map, [rows[0]])
    plt.close("all")

    with pytest.raises(ValueError, match="missing 'value'"):
        plot_embedding_map(embedding_map, [{**rows[0], "value": ""}, rows[1]], value_column="value")
    plt.close("all")

    with pytest.raises(ValueError, match="non-numeric 'value'"):
        plot_embedding_map(
            embedding_map,
            [{**rows[0], "value": "bad"}, rows[1]],
            value_column="value",
        )
    plt.close("all")

    with pytest.raises(ValueError, match="missing 'trace'"):
        plot_embedding_map(
            embedding_map,
            [{**rows[0], "trace": ""}, rows[1]],
            trace_column="trace",
            order_column="step",
        )
    plt.close("all")

    with pytest.raises(ValueError, match="missing 'step'"):
        plot_embedding_map(
            embedding_map,
            [{**rows[0], "step": ""}, rows[1]],
            trace_column="trace",
            order_column="step",
        )
    plt.close("all")

    with pytest.raises(ValueError, match="non-numeric 'value'"):
        plot_embedding_map(
            embedding_map,
            [{**rows[0], "value": "bad"}, rows[1]],
            trace_column="trace",
            order_column="step",
            value_column="value",
        )
    plt.close("all")

    with pytest.raises(ValueError, match="embedding_maps must contain"):
        plot_embedding_map_grid({}, rows)
    plt.close("all")
