.. hemolearn documentation master file, created by
   sphinx-quickstart on Sat Feb 20 19:09:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |logo| image:: _static/logo_hemolearn.png
  :width: 120
  :alt: HemoLearn logo


================
|logo| HemoLearn
================

Estimation of the Haemodnyamic Response Function from fMRI data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**HemoLearn** is a Python module to estimate the Haemodynamic Response Function (HRF) in brain from resting-state or task fMRI data (BOLD signal). It relies on a Sparse Low-Rank Deconvolution Analysis (SLRDA) to distangles the neurovascular coupling from the the neural activity.

|Python36|_ |Travis|_ |Codecov|_

.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
.. _Python36: https://badge.fury.io/py/scikit-learn

.. |Travis| image:: https://travis-ci.com/hemolearn/hemolearn.svg?branch=master
.. _Travis: https://travis-ci.com/hemolearn/hemolearn

.. |Codecov| image:: https://codecov.io/gh/hemolearn/hemolearn/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/hemolearn/hemolearn


Installation
~~~~~~~~~~~~

In order to perform the installation, run the following command from the hemolearn directory (from `GitHub repository <https://github.com/hemolearn/hemolearn>`_ )::

    python3 setup.py install --user


Contents
~~~~~~~~

.. toctree::
   :maxdepth: 1

   model
   auto_examples/index
   api
   bibliography


* :ref:`genindex`
* :ref:`search`
