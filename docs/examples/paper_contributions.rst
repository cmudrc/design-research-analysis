Paper Contributions
===================

Source: ``examples/paper_contributions.py``

Introduction
------------

Wrap an existing analysis result in a portable record that distinguishes executed
evidence from planned methods and produces restrained Methods and Results support.

Technical Implementation
------------------------

1. Fit a deterministic ordinary least-squares regression.
2. Record the hypothesis, included runs, documented exclusion, and diagnostic checks.
3. Write and reload the result beneath the study artifact directory.
4. Collect a JSON-compatible packet for downstream paper-draft assembly.

.. literalinclude:: ../../examples/paper_contributions.py
   :language: python
   :lines: 22-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/paper_contributions.py

Prints the two contract versions, the reloaded analysis identity, and the generated
contribution and reporting-gap identifiers. The gap makes the missing coefficient
uncertainty visible instead of implying a significance test.

References
----------

- docs/paper_contributions.rst
