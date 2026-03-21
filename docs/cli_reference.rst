CLI Reference
=============

The CLI is designed for deterministic pipeline integration. Each command writes
machine-readable summaries that can be versioned and compared across runs.

Entry point:

.. code-block:: bash

   design-research-analysis <subcommand> [options]

Output Envelope
---------------

All ``--summary-json`` outputs include common top-level keys:

- ``analysis``
- ``mode``
- ``output_schema_version``

Current schema version: ``1.0``.

Mapper Spec Format
------------------

For mapper options (such as ``--actor-mapper``), both formats are supported:

- ``module:function``
- ``module.function``

Subcommands
-----------

``validate-table``
~~~~~~~~~~~~~~~~~~

Validates schema compliance and writes a JSON report.

.. code-block:: bash

   design-research-analysis validate-table \
     --input data/events.csv \
     --summary-json artifacts/validate.json

``profile-dataset``
~~~~~~~~~~~~~~~~~~~

Profiles a dataset and writes summary diagnostics (missingness, inferred dtypes, and warnings).

.. code-block:: bash

   design-research-analysis profile-dataset \
     --input data/events.csv \
     --summary-json artifacts/dataset_profile.json \
     --max-categorical-levels 20

``validate-dataset``
~~~~~~~~~~~~~~~~~~~~

Validates a dataset against a JSON schema object.

.. code-block:: bash

   design-research-analysis validate-dataset \
     --input data/events.csv \
     --summary-json artifacts/dataset_validate.json \
     --schema-json '{"participant_id": {"unique": true, "nullable": false}}'

``generate-codebook``
~~~~~~~~~~~~~~~~~~~~~

Generates a codebook CSV plus a JSON summary.

.. code-block:: bash

   design-research-analysis generate-codebook \
     --input data/events.csv \
     --summary-json artifacts/codebook_summary.json \
     --codebook-csv artifacts/codebook.csv

``capture-context``
~~~~~~~~~~~~~~~~~~~

Captures runtime provenance context and optionally writes a manifest JSON file.

.. code-block:: bash

   design-research-analysis capture-context \
     --summary-json artifacts/context_summary.json \
     --manifest-json artifacts/run_manifest.json \
     --seed 7 \
     --input-path data/events.csv

``run-language``
~~~~~~~~~~~~~~~~

Runs semantic convergence plus sentiment analysis, and optionally topic modeling.

.. code-block:: bash

   design-research-analysis run-language \
     --input data/events.csv \
     --summary-json artifacts/language.json \
     --trajectory-csv artifacts/language_trajectory.csv \
     --include-topics \
     --n-topics 6

``run-embedding-maps``
~~~~~~~~~~~~~~~~~~~~~~

Runs embedding-map construction, clustering, and optional plotting/trajectory diagnostics.

.. code-block:: bash

   design-research-analysis run-embedding-maps \
     --input data/events.csv \
     --summary-json artifacts/embedding_maps.json \
     --map-csv artifacts/embedding_maps.csv \
     --method pca \
     --method umap \
     --trace-column session_id \
     --order-column timestamp \
     --comparison-png artifacts/embedding_maps.png

The summary JSON includes ``maps``, ``clustering``, ``coverage``, and
``trajectory`` blocks keyed by method name. Text-driven runs also include an
``embedding`` block, while feature-driven runs report the selected source
columns.

``run-sequence``
~~~~~~~~~~~~~~~~

Runs Markov or HMM sequence models from unified-table rows.

.. code-block:: bash

   design-research-analysis run-sequence \
     --input data/events.csv \
     --summary-json artifacts/sequence.json \
     --mode markov \
     --matrix-png artifacts/transition_matrix.png

``run-stats``
~~~~~~~~~~~~~

Runs group comparison, regression, or mixed-effects models.

.. code-block:: bash

   design-research-analysis run-stats \
     --input data/events.csv \
     --summary-json artifacts/stats.json \
     --mode regression \
     --x-columns x1,x2 \
     --y-column y
