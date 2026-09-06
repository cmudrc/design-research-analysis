Verified Paper Bundle
=====================

Source: ``examples/verified_paper_bundle.py``

Introduction
------------

Package a completed study handoff for collaborator inspection, manual
reanalysis, and continued paper editing without promising automatic replay or
environment recreation.

Technical Implementation
------------------------

1. Write a tiny canonical study fixture and one retained run record.
2. Persist one structured analysis result and a complete paper-draft tree.
3. Explicitly select one supporting-data table while leaving an attachment out.
4. Create the deterministic ZIP and verify every member hash without extraction.

.. literalinclude:: ../../examples/verified_paper_bundle.py
   :language: python
   :lines: 24-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/verified_paper_bundle.py

Prints a successful verification summary for ``study-paper-draft.zip``. The
archive contains canonical artifacts, run and analysis records, the paper
draft, sanitized environment metadata, and the explicitly selected support
file; the unselected attachment is absent.

References
----------

- docs/research_bundle.rst
