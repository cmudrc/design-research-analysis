Stats Interrater Reliability
============================

Source: ``examples/stats_interrater_reliability.py``

Introduction
------------

Protocol studies often begin with multiple researchers assigning nominal codes
to the same design moves. This example estimates agreement beyond chance using
Cohen's kappa, Fleiss' kappa, and Krippendorff's alpha.

Technical Implementation
------------------------

The coding matrix uses one row per protocol segment and one column per rater.
All three estimates use the same explicit nominal labels. A seeded item
bootstrap demonstrates the optional uncertainty interval without introducing
an external statistics dependency.

.. literalinclude:: ../../examples/stats_interrater_reliability.py
   :language: python
   :lines: 28-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/stats_interrater_reliability.py

The three coefficients are positive because most segments agree, but they are
below one because the raters disagree on two segments. Repeated runs produce
the same bootstrap intervals.

References
----------

Cohen (1960), Fleiss (1971), and Krippendorff (2011) define the reliability
coefficients demonstrated here.
