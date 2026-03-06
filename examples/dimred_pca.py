"""Project vectors with PCA and cluster the resulting coordinates."""

from __future__ import annotations

import numpy as np

from design_research_analysis import cluster_projection, reduce_dimensions


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
    projection = reduce_dimensions(vectors, method="pca", n_components=2)
    clusters = cluster_projection(
        projection.projection,
        method="kmeans",
        n_clusters=2,
        random_state=1,
    )
    print(projection.to_dict())
    print(clusters)


if __name__ == "__main__":
    main()
