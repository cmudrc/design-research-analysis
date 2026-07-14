"""Inter-rater reliability for coded design protocols.

## Introduction

Protocol studies often begin with multiple researchers assigning nominal codes
to the same design moves. This example estimates agreement beyond chance using
Cohen's kappa, Fleiss' kappa, and Krippendorff's alpha.

## Technical Implementation

The coding matrix uses one row per protocol segment and one column per rater.
All three estimates use the same explicit nominal labels. A seeded item
bootstrap demonstrates the optional uncertainty interval without introducing
an external statistics dependency.

## Expected Results

The three coefficients are positive because most segments agree, but they are
below one because the raters disagree on two segments. Repeated runs produce
the same bootstrap intervals.

## References

Cohen (1960), Fleiss (1971), and Krippendorff (2011) define the reliability
coefficients demonstrated here.
"""

from __future__ import annotations

import design_research_analysis as dran


def main() -> None:
    """Estimate three nominal agreement coefficients."""
    codings = [
        ["problem", "problem", "problem"],
        ["solution", "problem", "problem"],
        ["evaluation", "evaluation", "evaluation"],
        ["solution", "solution", "solution"],
        ["problem", "solution", "solution"],
        ["evaluation", "evaluation", "evaluation"],
    ]

    for method in ("cohen_kappa", "fleiss_kappa", "krippendorff_alpha"):
        method_codings = [row[:2] for row in codings] if method == "cohen_kappa" else codings
        result = dran.compute_interrater_reliability(
            method_codings,
            method=method,
            n_bootstrap=200,
            seed=17,
        )
        print(
            method,
            f"coefficient={result.coefficient:.3f}",
            f"interval={result.confidence_interval}",
        )


if __name__ == "__main__":
    main()
