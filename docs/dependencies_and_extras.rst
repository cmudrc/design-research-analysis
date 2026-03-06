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

- `seq`:
  - `hmmlearn`
  - `networkx`
  - `scipy`
- `embeddings`:
  - `sentence-transformers`

Install extras with:

.. code-block:: bash

   pip install -e ".[dev]"
   pip install -e ".[seq]"
   pip install -e ".[embeddings]"
