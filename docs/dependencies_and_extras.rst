Dependencies And Extras
=======================

The base install keeps required runtime dependencies intentionally small:

- `numpy`
- `matplotlib`

The `dev` extra installs the local contributor toolchain:

- `build`
- `mypy`
- `pre-commit`
- `pytest`
- `pytest-cov`
- `ruff`
- `sphinx`
- `sphinx-rtd-theme`
- `twine`
- `uv`

Optional feature extras:

- `table`:
  - (no additional dependencies; semantic marker for table-focused workflows)
- `data`:
  - `pandas`
- `seq`:
  - `hmmlearn`
  - `networkx`
  - `scipy`
- `embeddings`:
  - `sentence-transformers`
- `lang`:
  - `scikit-learn`
- `dimred`:
  - `scikit-learn`
  - `umap-learn`
- `stats`:
  - `scipy`
  - `statsmodels`
  - `pandas`
- `all`:
  - Installs all optional analysis extras.

Install extras with:

.. code-block:: bash

   pip install -e ".[dev]"
   pip install -e ".[table]"
   pip install -e ".[seq]"
   pip install -e ".[embeddings]"
   pip install -e ".[lang]"
   pip install -e ".[dimred]"
   pip install -e ".[data]"
   pip install -e ".[stats]"
   pip install -e ".[all]"
