# Examples

The examples in this repository are intentionally small but map to common
analysis workflows used in lab studies.

Sphinx example pages are generated directly from each example file's top module
docstring via ``scripts/generate_example_docs.py``.

- `basic_usage.py`: end-to-end unified table + Markov + language convergence for one session.
- `unified_table_validation.py`: normalize loose transcript rows into validated unified-table records.
- `sequence_from_table.py`: fit a Markov chain from event-coded session traces.
- `language_custom_embedder.py`: run convergence analysis with deterministic in-house embeddings.
- `dimred_pca.py`: PCA projection and k-means clustering over study-level vectors.
- `stats_regression.py`: novelty-vs-iteration regression for prototype runs.
- `lab_study_pipeline.py`: prompt-framing experiment pipeline with table checks, language/sequence/stats, and provenance manifest output.

All examples use the import convention:

```python
import design_research_analysis as dran
```

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

For example:

```bash
PYTHONPATH=src python examples/lab_study_pipeline.py
```
