# Examples

The examples in this repository are intentionally small but map to common
analysis workflows used in lab studies.

Sphinx example pages are generated directly from each example file's top module
docstring via ``scripts/generate_example_docs.py``.

- `basic_usage.py`: end-to-end unified table + Markov comparison + language convergence for one session.
- `unified_table_validation.py`: normalize loose transcript rows into validated unified-table records.
- `sequence_from_table.py`: fit a Markov chain from event-coded session traces.
- `language_custom_embedder.py`: run convergence analysis with deterministic in-house embeddings.
- `embedding_maps_trajectories.py`: trace-aware embedding maps with value overlays and comparison grids.
- `idea_space_metrics.py`: projection-space coverage, trajectory diagnostics, and plotting helpers.
- `mechanical_design_review_analysis.py`: combine process, condition-metric, language, and visualization helpers for a bracket design review.
- `stats_regression.py`: novelty-vs-iteration regression for prototype runs.
- `paper_contributions.py`: portable analysis records and evidence-linked paper support.
- `verified_paper_bundle.py`: deterministic, integrity-checked paper-draft handoff with narrow data selection.
- `stats_interrater_reliability.py`: Cohen, Fleiss, and nominal Krippendorff reliability for protocol codings.
- `condition_pair_significance.py`: join canonical experiment exports into run-level metrics and render pairwise significance summaries.
- `experiment_artifacts_handoff.py`: artifact-first condition, sequence, and regression analyses over canonical experiment exports.
- `lab_study_pipeline.py`: prompt-framing experiment pipeline with table checks, language/sequence/stats, and provenance manifest output.

All examples use the import convention:

```python
import design_research_analysis as dran
```

Run the full example suite:

```bash
python -m pip install -e ".[dev,data,stats,lang,maps,seq]"
make run-examples
```

Several individual scripts use only the base install, but the full suite
intentionally covers optional families. Text embeddings in these examples use
deterministic custom embedders, so the suite does not download a model.

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
