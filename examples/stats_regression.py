"""Fit an ordinary least squares regression model."""

from __future__ import annotations

from design_research_analysis import fit_regression


def main() -> None:
    """Run and print a small linear model."""
    X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    y = [1.0, 3.0, 5.0, 7.0, 9.0]
    result = fit_regression(X, y, feature_names=["x"])
    print(result.to_dict())


if __name__ == "__main__":
    main()
