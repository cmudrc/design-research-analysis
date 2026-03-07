"""Project ideation vectors with PCA and cluster session-level coordinates.

## Introduction
Reduce higher-dimensional study vectors to two dimensions for plotting and then
cluster projected points into interpretable groups.

## Technical Implementation
1. Build a deterministic in-memory numeric matrix.
2. Run PCA projection with two components.
3. Run k-means clustering on projected coordinates.

## Expected Results
Prints a projection dictionary (shape, method, variance ratio) and a clustering
dictionary with labels, centers, and config.

## References
- docs/analysis_recipes.rst
"""

from __future__ import annotations

import numpy as np

import design_research_analysis as dran


def main() -> None:
    """Run PCA and k-means clustering on toy vectors."""
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.1],
            [0.9, 0.2, 0.1],
            [0.0, 1.0, 0.2],
            [0.1, 0.9, 0.3],
            [0.5, 0.5, 0.8],
        ]
    )
    projection = dran.reduce_dimensions(vectors, method="pca", n_components=2)
    clusters = dran.cluster_projection(
        projection.projection,
        method="kmeans",
        n_clusters=2,
        random_state=1,
    )
    print(projection.to_dict())
    print(clusters)


if __name__ == "__main__":
    main()
