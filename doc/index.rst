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

**HemoLearn** is a Python module offering a new algorithm that aims to fit a
rich multivariate decomposition of the BOLD data using a semi-blind
deconvolution and low-rank sparse decomposition. The model distinguishes two
major parts in the BOLD signal: the neurovascular coupling and the neural
activity signal.

|Travis|_ |Codecov|_

.. |Travis| image:: https://app.travis-ci.com/hemolearn/hemolearn.svg?branch=master
.. _Travis: https://app.travis-ci.com/hemolearn/hemolearn

.. |Codecov| image:: https://codecov.io/gh/hemolearn/hemolearn/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/hemolearn/hemolearn


Installation
~~~~~~~~~~~~

In order to perform the installation, run the following command from the hemolearn directory (obtained from `GitHub repository <https://github.com/hemolearn/hemolearn>`_ )::

    python3 setup.py install --user


Contents
~~~~~~~~

.. toctree::
   :maxdepth: 1

   model
   auto_examples/index
   api
   bibliography


Navigate
~~~~~~~~

* :ref:`genindex`
* :ref:`search`
