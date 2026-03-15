# design-research-analysis
[![CI](https://github.com/cmudrc/design-research-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/cmudrc/design-research-analysis/actions/workflows/ci.yml)
[![Docs](https://github.com/cmudrc/design-research-analysis/actions/workflows/docs-pages.yml/badge.svg)](https://github.com/cmudrc/design-research-analysis/actions/workflows/docs-pages.yml)

<!-- release-callout:start -->
> [!IMPORTANT]
> Current monthly release: [Allegheny Analysis - April 2026](https://github.com/cmudrc/design-research-analysis/milestone/1)  
> Due: April 1, 2026  
> Tracks: March 2026 work
<!-- release-callout:end -->

`design-research-analysis` is the unified-table analysis layer in the cmudrc design research ecosystem.

It provides typed, reusable workflows for sequence, language, embedding-map, and statistical analysis over recurring event logs.

## Overview

This package centers on reproducible analysis workflows with a small top-level API:

- Unified-table coercion, validation, and mapper-based derived columns
- Dataset profiling, schema checks, and codebook generation
- Sequence modeling (Markov chains, discrete HMM, Gaussian HMM)
- Language analysis (semantic convergence trajectories, topic modeling, sentiment scoring)
- Embedding maps (PCA, t-SNE, UMAP, PaCMAP, TriMap) with clustering and trajectory-plotting helpers
- Statistical wrappers (group comparisons, OLS regression, mixed-effects models, nonparametrics, and power)
- Runtime provenance capture for reproducibility manifests
- A thin CLI for deterministic pipeline runs

## Quickstart

Requires Python 3.12+.
Reproducible release installs are pinned to Python `3.12.12` (`.python-version`).

```bash
python -m venv .venv
source .venv/bin/activate
make dev
make test
```

Run a compact end-to-end example:

```bash
PYTHONPATH=src python examples/basic_usage.py
```

For frozen installs and release-check guidance, see [Dependencies and Extras](https://cmudrc.github.io/design-research-analysis/dependencies_and_extras.html).

## CLI

The package installs a `design-research-analysis` CLI:

```bash
design-research-analysis validate-table --input data/events.csv --summary-json artifacts/validate.json
design-research-analysis run-sequence --input data/events.csv --summary-json artifacts/sequence.json --mode markov
design-research-analysis run-language --input data/events.csv --summary-json artifacts/language.json --trajectory-csv artifacts/language_trajectory.csv
design-research-analysis run-embedding-maps --input data/events.csv --summary-json artifacts/embedding_maps.json --map-csv artifacts/embedding_maps.csv
design-research-analysis run-stats --input data/events.csv --summary-json artifacts/stats.json --mode regression --x-columns x1,x2 --y-column y
```

## Examples

Start with [examples/README.md](https://github.com/cmudrc/design-research-analysis/blob/main/examples/README.md) for runnable scripts across all analysis families.

## Docs

See the [published documentation](https://cmudrc.github.io/design-research-analysis/) for quickstart, workflow guidance, schema details, CLI reference, and API docs.

Build docs locally with:

```bash
make docs
```

## Public API

The supported public surface is whatever is exported from `design_research_analysis.__all__`.

Top-level exports include:

- Table contracts: `UnifiedTableConfig`, `UnifiedTableValidationReport`, `coerce_unified_table`, `derive_columns`, `validate_unified_table`
- Sequence: `fit_markov_chain_from_table`, `fit_discrete_hmm_from_table`, `fit_text_gaussian_hmm_from_table`, `decode_hmm`, plotting helpers, and result types
- Language: `compute_language_convergence`, `compute_semantic_distance_trajectory`, `fit_topic_model`, `score_sentiment`
- Embedding maps: `embed_records`, `build_embedding_map`, `cluster_embedding_map`, `compare_embedding_maps`, `plot_embedding_map`, `plot_embedding_map_grid`
- Statistics: `compare_groups`, `fit_regression`, `fit_mixed_effects`, `permutation_test`, `bootstrap_ci`, power helpers
- Dataset + runtime: `profile_dataframe`, `validate_dataframe`, `generate_codebook`, `capture_run_context`, `attach_provenance`, `write_run_manifest`

## Contributing

Contribution workflow and validation gates are documented in [CONTRIBUTING.md](https://github.com/cmudrc/design-research-analysis/blob/main/CONTRIBUTING.md).
