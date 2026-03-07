"""Fit a simple novelty-vs-iteration regression for lab study outputs.

## Introduction
Estimate a linear trend between prototype iteration index and novelty score for a
compact design-study sample.

## Technical Implementation
1. Define one explanatory feature (iteration count).
2. Fit an OLS regression through the package helper.
3. Print the serialized coefficient and fit diagnostics.

## Expected Results
Prints regression coefficients, intercept, R2, MSE, and input-shape metadata.

## References
- docs/analysis_recipes.rst
"""

from __future__ import annotations

import design_research_analysis as dran


def main() -> None:
    """Run and print a small linear model."""
    prototype_iteration = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    novelty_score = [1.0, 3.0, 5.0, 7.0, 9.0]
    result = dran.fit_regression(
        prototype_iteration,
        novelty_score,
        feature_names=["prototype_iteration"],
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()
