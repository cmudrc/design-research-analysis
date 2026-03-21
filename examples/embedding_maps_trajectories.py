"""Plot trace-aware embedding maps with scalar value overlays.

## Introduction
Map ordered design traces into a shared coordinate space, then overlay their
paths and scalar values so exploration behavior is visible instead of just the
point cloud.

## Technical Implementation
1. Build a tiny unified-table-like dataset with record IDs, trace IDs, steps,
   text, and performance values.
2. Embed the text deterministically with a local lookup and build a PCA
   embedding map.
3. Render a single value-colored trajectory map and a comparison grid; when no
   optional mapping backend is installed, compare PCA with a reflected PCA map
   so the plotting workflow still runs end to end.

## Expected Results
Writes ``artifacts/examples/embedding_map.png`` and
``artifacts/examples/embedding_map_grid.png``, then prints the method names
included in the comparison.

## References
- docs/workflows_embedding_maps.rst
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

import design_research_analysis as dran


def _local_embed(texts: list[str]) -> np.ndarray:
    lookup = {
        "seed concept": [0.0, 1.0, 0.0],
        "refined concept": [0.3, 0.9, 0.2],
        "prototype concept": [0.8, 0.4, 0.4],
        "alternate seed": [0.1, 0.2, 1.0],
        "alternate refinement": [0.2, 0.1, 0.8],
        "alternate prototype": [0.7, 0.3, 0.5],
    }
    return np.asarray([lookup[text] for text in texts], dtype=float)


def _comparison_methods() -> list[str]:
    if importlib.util.find_spec("umap") is not None:
        return ["pca", "umap"]
    if importlib.util.find_spec("sklearn") is not None:
        return ["pca", "tsne"]
    if importlib.util.find_spec("pacmap") is not None:
        return ["pca", "pacmap"]
    if importlib.util.find_spec("trimap") is not None:
        return ["pca", "trimap"]
    return ["pca"]


def main() -> None:
    """Build and plot trajectory-aware embedding maps."""
    rows = [
        {
            "record_id": "r1",
            "trace_id": "alpha",
            "step": 1,
            "text": "seed concept",
            "value": 0.2,
        },
        {
            "record_id": "r2",
            "trace_id": "alpha",
            "step": 2,
            "text": "refined concept",
            "value": 0.5,
        },
        {
            "record_id": "r3",
            "trace_id": "alpha",
            "step": 3,
            "text": "prototype concept",
            "value": 0.9,
        },
        {
            "record_id": "r4",
            "trace_id": "beta",
            "step": 1,
            "text": "alternate seed",
            "value": 0.1,
        },
        {
            "record_id": "r5",
            "trace_id": "beta",
            "step": 2,
            "text": "alternate refinement",
            "value": 0.4,
        },
        {
            "record_id": "r6",
            "trace_id": "beta",
            "step": 3,
            "text": "alternate prototype",
            "value": 0.8,
        },
    ]

    embedded = dran.embed_records(rows, embedder=_local_embed)
    pca_map = dran.build_embedding_map(
        embedded.embeddings,
        method="pca",
        record_ids=embedded.record_ids,
    )

    methods = _comparison_methods()
    if len(methods) > 1:
        comparison_maps = dran.compare_embedding_maps(
            embedded.embeddings,
            methods=methods,
            record_ids=embedded.record_ids,
            n_neighbors=3,
            perplexity=2.0,
        )
    else:
        comparison_maps = {
            "pca": pca_map,
            "pca_reflected": dran.EmbeddingMapResult(
                coordinates=np.column_stack(
                    [pca_map.coordinates[:, 0], -pca_map.coordinates[:, 1]]
                ),
                record_ids=list(pca_map.record_ids),
                method="pca_reflected",
                config={"derived_from": "pca"},
            ),
        }

    output_dir = Path("artifacts/examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    single_path = output_dir / "embedding_map.png"
    grid_path = output_dir / "embedding_map_grid.png"

    figure, _ = dran.plot_embedding_map(
        pca_map,
        rows,
        trace_column="trace_id",
        order_column="step",
        value_column="value",
        title="PCA trajectory map",
    )
    figure.savefig(single_path, dpi=150, bbox_inches="tight")
    figure.clf()

    grid_figure, _ = dran.plot_embedding_map_grid(
        comparison_maps,
        rows,
        trace_column="trace_id",
        order_column="step",
        value_column="value",
    )
    grid_figure.savefig(grid_path, dpi=150, bbox_inches="tight")
    grid_figure.clf()

    print(f"Single map: {single_path}")
    print(f"Comparison grid: {grid_path}")
    print(f"Methods: {list(comparison_maps)}")


if __name__ == "__main__":
    main()
