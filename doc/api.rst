.. _api_documentation:

API Documentation
=================

.. currentmodule:: hemolearn

Sparse Low Rank Decomposition Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   hemolearn.SLRDA
   hemolearn.deconvolution.blind_deconvolution_multiple_subjects
   hemolearn.deconvolution.multi_runs_blind_deconvolution_multiple_subjects

Vascular atlas
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   hemolearn.atlas.fetch_basc_vascular_atlas
   hemolearn.atlas.fetch_harvard_vascular_atlas
   hemolearn.atlas.fetch_aal_vascular_atlas
   hemolearn.atlas.fetch_aal3_vascular_atlas


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

      hemolearn.plotting.plot_vascular_map
      hemolearn.plotting.plot_spatial_maps
      hemolearn.plotting.plot_temporal_activations

Utilities functions
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

      hemolearn.utils.sort_by_expl_var
      hemolearn.utils.fwhm
      hemolearn.utils.tp
