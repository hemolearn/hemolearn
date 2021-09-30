.. _api_documentation:

API Documentation
=================

.. currentmodule:: hemolearn

Sparse Low Rank Decomposition Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   hemolearn.SLRDA
   hemolearn.learn_u_z_v_multi.learn_u_z_v_multi
   hemolearn.learn_u_z_v_multi.multi_runs_learn_u_z_v_multi

Vascular atlas
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   hemolearn.atlas.fetch_atlas_basc_2015
   hemolearn.atlas.fetch_vascular_atlas


Generating synthesis data
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

      hemolearn.simulated_data.simulated_data

Convolution functions
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

      hemolearn.convolution.make_toeplitz

Haemodynamic Response Function models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

      hemolearn.hrf_model.hrf_2_basis
      hemolearn.hrf_model.hrf_3_basis
      hemolearn.hrf_model.scaled_hrf

Plotting functions
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

      hemolearn.plotting.plotting_hrf
      hemolearn.plotting.plotting_hrf_stats
      hemolearn.plotting.plotting_obj_values
      hemolearn.plotting.plotting_spatial_comp
      hemolearn.plotting.plotting_temporal_comp

Utilities functions
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

      hemolearn.utils.fmri_preprocess
      hemolearn.utils.sort_atoms_by_explained_variances
