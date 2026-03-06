CLI Reference
=============

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

``run-dimred``
~~~~~~~~~~~~~~

Runs embedding, projection, and clustering.

.. code-block:: bash

   design-research-analysis run-dimred \
     --input data/events.csv \
     --summary-json artifacts/dimred.json \
     --projection-csv artifacts/projection.csv \
     --method pca \
     --n-components 2 \
     --n-clusters 3

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
