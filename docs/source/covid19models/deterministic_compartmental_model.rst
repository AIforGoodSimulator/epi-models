Deterministic Compartmental Model
=================================

Summary
=======
This documentation elaborates on the compartmental model used by the `Crisis Modelling team <https://www.aiforgoodsimulator.com/>`_ to forcast the spread of the COVID'19 disease. We will first understand to mathematically model the pandemic using ordinary differential equations and then use python to simulate the model.  Later we visualize the effects of the spread like the one in the figure below.

Compartmental Model
==========
The model comprises of 11 disease state compartments and 8 age compartments with a 10 years gap, starting from 0-10 to 70 and above.

.. figure:: Compartments.png
 :scale: 50 %
 :alt: map to buried treasure

  Model Diagram.

Define the ODEs
==========
The ordinary differential equations(ODEs) for the compartmental model are defined as follows:

.. code-block:: python

    S_vec=y2d[Config.compartment_index["S"],:]      
    # (-scenario_dict["transmission_reduction_factor"]* beta* S_vec* infection_total- offsite)
    E_vec = y2d[Config.compartment_index["E"],:]
    # (-scenario_dict["transmission_reduction_factor"]* beta* S_vec* infection_total- E_latent)
    I_vec = y2d[Config.compartment_index["I"], :]
    # (self.p_symptomatic*E_latent-I_removed- quarantine_sicks+quarantined_sicks_sendback)
    H_vec = y2d[Config.compartment_index["H"], :]
    #  (self.p_hosp_given_symptomatic * I_removed- hosp_rate * H_vec+ death_rate_ICU*(1-self.death_prob_with_ICU) * np.minimum(C_vec, hospitalized_on_icu)+self.p_hosp_given_symptomatic * Q_quarantined)
    A_vec = y2d[Config.compartment_index["A"], :]
    # (1 - self.p_symptomatic) * E_latent - A_removed)
    C_vec = y2d[Config.compartment_index["C"], :]
    # (icu_cared - deaths_on_icu)
    Q_vec = y2d[Config.compartment_index["Q"], :]
    # (quarantine_sicks - Q_quarantined - quarantined_sicks_sendback)
    dydt2d[Config.compartment_index["D"], :] = (deaths_without_icu+ self.death_prob_with_ICU* deaths_on_icu)
    dydt2d[Config.compartment_index["O"], :] = offsite

Simulating the model
==========
...

Visualization
==========
...

Conclusion
==========
...

References
==========
...



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



