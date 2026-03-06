# design-research-analysis

`design-research-analysis` is a typed Python package for recurring design-research analyses over a unified event table.

## What It Includes

- Unified-table coercion, validation, and mapper-based derived columns
- Dataset profiling, schema validation, and codebook generation
- Sequence modeling (Markov chains, discrete HMM, Gaussian HMM)
- Language analysis (semantic convergence trajectories, topic modeling, sentiment scoring)
- Embedding and dimensionality reduction (PCA, t-SNE, UMAP)
- Statistical wrappers (group comparisons, OLS regression, mixed-effects models, nonparametrics, power)
- Runtime provenance helpers (environment detection and reproducibility manifests)
- Thin CLI for reproducible pipeline runs

## Install

Base install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[seq]"
pip install -e ".[embeddings]"
pip install -e ".[lang]"
pip install -e ".[dimred]"
pip install -e ".[data]"
pip install -e ".[stats]"
pip install -e ".[all]"
```

Contributor toolchain:

```bash
pip install -e ".[dev]"
make test
```

## Unified Table Contract

Required column:

- `timestamp`

Strongly recommended columns:

- `record_id`
- `text`
- `session_id`
- `actor_id`
- `event_type`

Optional common column:

- `meta_json`

Loose schemas are supported. Missing sequence fields can be derived with deterministic mapper functions.

## Minimal API Example

```python
from design_research_analysis import (
    compute_language_convergence,
    fit_markov_chain_from_table,
    validate_unified_table,
)

rows = [
    {"timestamp": "2026-01-01T10:00:00Z", "session_id": "s1", "event_type": "A", "text": "alpha"},
    {"timestamp": "2026-01-01T10:00:01Z", "session_id": "s1", "event_type": "B", "text": "beta"},
    {"timestamp": "2026-01-01T10:00:02Z", "session_id": "s1", "event_type": "A", "text": "gamma"},
]

report = validate_unified_table(rows)
assert report.is_valid

markov = fit_markov_chain_from_table(rows)
print(markov.transition_matrix)

embedding_lookup = {
    "alpha": [2.0, 0.0],
    "beta": [1.0, 0.0],
    "gamma": [0.0, 0.0],
}

convergence = compute_language_convergence(
    rows,
    window_size=1,
    embedder=lambda texts: [embedding_lookup[text] for text in texts],
)
print(convergence.direction_by_group)
```

## CLI

Validate input:

```bash
design-research-analysis validate-table \
  --input data/events.csv \
  --summary-json artifacts/validate.json
```

Run sequence analysis:

```bash
design-research-analysis run-sequence \
  --input data/events.csv \
  --summary-json artifacts/sequence.json \
  --mode markov
```

Run language analysis:

```bash
design-research-analysis run-language \
  --input data/events.csv \
  --summary-json artifacts/language.json \
  --trajectory-csv artifacts/language_trajectory.csv
```

Profile a dataset:

```bash
design-research-analysis profile-dataset \
  --input data/events.csv \
  --summary-json artifacts/dataset_profile.json
```

Capture runtime provenance context:

```bash
design-research-analysis capture-context \
  --summary-json artifacts/context_summary.json \
  --manifest-json artifacts/run_manifest.json \
  --seed 7 \
  --input-path data/events.csv
```

## More Examples

Additional runnable scripts are in `examples/`:

- `examples/unified_table_validation.py`
- `examples/sequence_from_table.py`
- `examples/language_custom_embedder.py`
- `examples/dimred_pca.py`
- `examples/stats_regression.py`
- `examples/api_surface_walkthrough.py`

Run the full example suite:

```bash
make run-examples
```

Check public API coverage across examples:

```bash
make examples-coverage
```

Run any example with:

```bash
PYTHONPATH=src python examples/<example_name>.py
```

## Documentation

Sphinx pages now include:

- `quickstart`
- `workflows`
- `unified_table_schema`
- `cli_reference`
- `analysis_recipes`
- `dependencies_and_extras`
- `api`

## Development Commands

- `make fmt`
- `make lint`
- `make type`
- `make test`
- `make run-example`
- `make run-examples`
- `make examples-coverage`
- `make ci`
