Verified Paper Bundle
=====================

``design-research-analysis`` can package a completed experiment handoff into a
deterministic ``study-paper-draft.zip`` for integrity verification,
collaborator inspection, manual reanalysis from retained observations, and
continued paper-draft editing.

This is deliberately an inspection-oriented research artifact. It does not
execute code from the bundle and does not promise automatic analysis replay,
environment recreation, byte-identical model reruns, or regeneration of
live-model observations. It is not a storage or experiment-tracking platform.

Create and Verify
-----------------

.. code-block:: python

   import design_research_analysis as dran

   bundle = dran.create_research_bundle(
       "study-output/manifest.json",
       supporting_data=(
           "artifacts/analysis/supporting-data/analysis-input.csv",
       ),
   )
   verification = dran.verify_research_bundle(bundle)
   assert verification["valid"] is True

The default output is ``study-output/study-paper-draft.zip``. Existing output
is never replaced unless ``overwrite=True`` is explicit.

Included Content
----------------

The bundle root is ``study-paper-draft/`` and contains:

- canonical study artifacts: ``study.yaml``, ``manifest.json``, conditions,
  runs, events, evaluations, hypotheses, and the analysis plan;
- each per-run ``run.json`` plus ``observations.jsonl`` when present;
- validated records beneath ``artifacts/analysis/results/``;
- analysis tables and figures;
- the complete ``paper-draft/`` tree, including empty table/figure folders;
- optional ``component_metadata.json`` and ``analysis_results.json``;
- explicitly selected supporting-data files or directories; and
- ``bundle/environment.json`` with Python, platform, and allowlisted package
  versions.

Run attachments, unselected supporting data, environment variables, hostnames,
and executable paths are excluded by default. A sensitive file is included only
when the caller names its relative path through ``supporting_data``.

Integrity Model
---------------

``bundle_manifest.json`` inventories every other archive member with its type,
category, byte size, and SHA-256 digest. The manifest's own SHA-256 digest is
stored in the ZIP comment, avoiding a circular self-hash while covering every
member. ``verify_research_bundle`` checks:

- one safe ``study-paper-draft/`` archive root;
- no duplicate, absolute, traversal, or symlink members;
- the manifest digest from the ZIP comment;
- exact agreement between members and inventory; and
- every recorded size and SHA-256 hash.

Verification reads the archive directly and never extracts or executes it.

Source Validation
-----------------

Creation fails clearly when canonical artifacts, run records, analysis-result
records, paper-draft files, or analysis-referenced tables/figures are absent.
All selected source paths must remain beneath the study root and cannot traverse
symlinks. The output ZIP cannot be placed inside a selected source directory.

The archive uses stable member ordering, fixed ZIP timestamps and permissions,
and deterministic JSON serialization. Repeating creation with the same source
bytes and installed package versions produces identical archive bytes.
