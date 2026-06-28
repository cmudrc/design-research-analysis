Run An Example In VS Code
=========================

Use this page when you want to try ``design-research-analysis`` in VS Code.
Choose the installed-package path for a first user workflow, or the source
checkout path when you want to run the repository's checked-in examples and
development checks.

The checked-in ``examples/`` directory lives in the repository source. Do not
assume those files are present inside the PyPI wheel.

Requirements
------------

- Python 3.12 or newer. Maintainer workflows target the version in
  ``.python-version``.
- VS Code with the Python extension.
- A VS Code integrated terminal.

Installed Package From PyPI
---------------------------

Open an empty folder in VS Code, then create and activate a virtual
environment from ``Terminal > New Terminal``.

On macOS or Linux:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install design-research-analysis

On Windows PowerShell:

.. code-block:: powershell

   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install design-research-analysis

Run ``Python: Select Interpreter`` from the command palette and choose the
interpreter inside ``.venv``. If VS Code does not list it, enter the interpreter
path manually:

- macOS/Linux: ``.venv/bin/python``
- Windows: ``.venv\Scripts\python.exe``

Create ``analysis_example.py`` in the workspace folder:

.. code-block:: python

   from design_research_analysis import (
       coerce_unified_table,
       fit_markov_chain_from_table,
       validate_unified_table,
   )

   rows = coerce_unified_table(
       [
           {"timestamp": "2026-01-01T00:00:00", "session_id": "s1", "actor_id": "a", "event_type": "ideate"},
           {"timestamp": "2026-01-01T00:00:05", "session_id": "s1", "actor_id": "a", "event_type": "refine"},
           {"timestamp": "2026-01-01T00:00:10", "session_id": "s1", "actor_id": "a", "event_type": "evaluate"},
       ]
   )

   report = validate_unified_table(rows)
   if not report.is_valid:
       raise RuntimeError("; ".join(report.errors))

   result = fit_markov_chain_from_table(rows)
   print(result.states)

Run the file with VS Code's ``Run Python File`` action, or run:

.. code-block:: bash

   python analysis_example.py

Source Checkout For Repository Examples
---------------------------------------

Use this path when you want the checked-in examples, docs, tests, and optional
development tooling.

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-analysis.git
   cd design-research-analysis
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -e ".[dev]"

Equivalent maintainer shortcut:

.. code-block:: bash

   make dev

Run the deterministic example path from the integrated terminal:

.. code-block:: bash

   make run-example
   make examples-test
   python examples/basic_usage.py

First Development Checks
------------------------

Run the checks from VS Code's integrated terminal:

.. code-block:: bash

   make test
   make qa
   make docs-check

``make qa`` runs linting, formatting checks, type checks, and tests. Run
``make coverage`` before merge when changing tested behavior.

Optional Backends
-----------------

Install optional extras only when a workflow needs them:

.. code-block:: bash

   python -m pip install -e ".[seq]"
   python -m pip install -e ".[lang,embeddings]"
   python -m pip install -e ".[maps]"
   python -m pip install -e ".[stats,data]"

Runtime Caches
--------------

The Makefile keeps plotting and package caches under ``artifacts/runtime`` for
local analysis workflows. The relevant environment variables are
``MPLCONFIGDIR``, ``XDG_CACHE_HOME``, and ``MPLBACKEND``. Use
``make runtime-cache`` if those directories need to be created manually.

Troubleshooting
---------------

- If VS Code imports fail but the terminal works, reselect the ``.venv``
  interpreter and reload the window.
- If ``make`` uses the wrong Python, activate ``.venv`` in the terminal or run
  ``PYTHON=.venv/bin/python make test``.
- If Windows activation is blocked, switch the terminal profile to Command
  Prompt and run ``.\.venv\Scripts\activate.bat``.
- If a method-specific import is missing, install the matching optional extra
  rather than ``all`` unless you need a broad local analysis workstation.
- Avoid committing generated runtime output under ``artifacts/``,
  ``docs/_build/``, or local virtual environment directories.
