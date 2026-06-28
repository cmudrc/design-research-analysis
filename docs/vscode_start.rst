VS Code Start
=============

Use this path when opening ``design-research-analysis`` in VS Code for the first
time. The commands are repo-local and assume Python 3.12 or newer.

Requirements
------------

- Python 3.12 or newer. Maintainer workflows target the version in
  ``.python-version``.
- VS Code with the Python extension.
- ``make`` available in the integrated terminal.
- Optional: ``uv`` for faster virtual environment and package installs.

Open the Repository
-------------------

Open the repository root folder in VS Code, not the parent ecosystem folder. The
root should contain ``pyproject.toml``, ``Makefile``, ``src/``, ``tests/``, and
``examples/``.

Create an Environment
---------------------

Standard library setup:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

On Windows PowerShell, use:

.. code-block:: powershell

   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1

With ``uv``:

.. code-block:: bash

   uv venv --python 3.12
   source .venv/bin/activate

Install Development Dependencies
--------------------------------

Use the maintainer shortcut:

.. code-block:: bash

   make dev

Equivalent ``pip`` command:

.. code-block:: bash

   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -e ".[dev]"

Equivalent ``uv`` command:

.. code-block:: bash

   uv pip install -e ".[dev]"

Install optional extras only when a workflow needs them:

.. code-block:: bash

   python -m pip install -e ".[seq]"
   python -m pip install -e ".[lang,embeddings]"
   python -m pip install -e ".[maps]"
   python -m pip install -e ".[stats,data]"

Select the VS Code Interpreter
------------------------------

Run ``Python: Select Interpreter`` from the command palette and choose the
interpreter inside ``.venv``. If VS Code does not list it, enter the interpreter
path manually:

- macOS/Linux: ``.venv/bin/python``
- Windows: ``.venv\Scripts\python.exe``

First Checks
------------

Run the checks from VS Code's integrated terminal:

.. code-block:: bash

   make test
   make qa
   make docs-check

``make qa`` runs linting, formatting checks, type checks, and tests. Run
``make coverage`` before merge when changing tested behavior.

Deterministic Examples
----------------------

Run the compact deterministic example path from the integrated terminal:

.. code-block:: bash

   make run-example
   make examples-test

To run the primary script directly:

.. code-block:: bash

   PYTHONPATH=src python examples/basic_usage.py

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
- If Windows activation is blocked, enable script execution for the current
  user or use the VS Code Python extension's environment activation.
- If a method-specific import is missing, install the matching optional extra
  rather than ``all`` unless you need a broad local analysis workstation.
- Avoid committing generated runtime output under ``artifacts/``,
  ``docs/_build/``, or local virtual environment directories.
