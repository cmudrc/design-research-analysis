Sequence Workflows
==================

Use sequence workflows when temporal order and transition dynamics are central
study signals.

Typical Questions
-----------------

- How do participants move between event states?
- Are transition structures different across conditions?
- Do latent states explain observed trajectories?

Key API Entry Points
--------------------

- :func:`design_research_analysis.fit_markov_chain_from_table`
- :func:`design_research_analysis.fit_discrete_hmm_from_table`
- :func:`design_research_analysis.fit_text_gaussian_hmm_from_table`
- :func:`design_research_analysis.decode_hmm`

CLI Path
--------

.. code-block:: bash

   design-research-analysis run-sequence \
     --input data/events.csv \
     --summary-json artifacts/sequence.json \
     --mode markov
