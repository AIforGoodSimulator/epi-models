Epidemiology models by Crisis Modelling Documentation
=====================================================

Summary
=======
This repository contains the models adopted by the `Crisis Modelling team <https://www.aiforgoodsimulator.com/>`_ for infectious disease modelling in Humanitarian Settings (in particular refugee/IDP camps) and first of which for COVID-19 modelling starting from April 2020.
Here the models are brought together under standardised API and open sourced for others to better model infectious diseases in different settings they operate in. The Crisis Modelling team uses this package in the Simulator web tool that they develop for other humanitarian actors to use.

Installation
============
The package is published on pypi so::

    pip install simulator-epi-models

is sufficient.

Quickstart
==========
As of version 0.1.3, only deterministic compartmental model for COVID-19 modelling is ready for use and the package is shipped with an example JSON that you can try your hands on.
You can also check out the `demo notebook <https://github.com/AIforGoodSimulator/epi-models/blob/main/compartmental_model_demo.ipynb>`_ on how to use this model.

.. code-block:: python

    from epi_models import DeterministicCompartmentalModelRunner, CampParams
    import importlib.resources as pkg_resources
    camp_params = CampParams(pkg_resources.read_text(epi_models.config, "sample_input.json"))
    runner = DeterministicCompartmentalModelRunner(camp_params, num_iterations=10)
    do_nothing_baseline, camp_baseline = runner.run_baselines()
    better_hygiene_intervention_result, increase_icu_intervention_result, increase_remove_high_risk_result, \
    better_isolation_intervention_result, shielding_intervention_result = runner.run_different_scenarios()


Disclaimer
==========

The models are still underdevelopment and the official 1.0 release will mark the thorough testing of the model for portable use.

.. toctree::
   :maxdepth: 2

   covid19models/deterministic_compartmental_model



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
