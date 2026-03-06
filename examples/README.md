# Examples

The examples in this repository are intentionally small but cover all core
analysis families.

- `basic_usage.py`: end-to-end unified table + Markov + language convergence.
- `unified_table_validation.py`: loose schema normalization and validation.
- `sequence_from_table.py`: Markov chain fitting from event rows.
- `language_custom_embedder.py`: language convergence with deterministic embeddings.
- `dimred_pca.py`: PCA projection and k-means clustering.
- `stats_regression.py`: OLS regression wrapper usage.

Run any example with:

```bash
PYTHONPATH=src python examples/<example_name>.py
```

For example:

```bash
PYTHONPATH=src python examples/dimred_pca.py
```
