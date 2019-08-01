.. -*- mode: rst -*-

|Python35|_ |Travis|_ |Codecov|_


.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/scikit-learn

.. |Travis| image:: https://travis-ci.com/CherkaouiHamza/seven.svg?branch=master
.. _Travis: https://travis-ci.com/CherkaouiHamza/seven


.. |Codecov| image:: https://codecov.io/gh/CherkaouiHamza/seven/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/CherkaouiHamza/seven


Seven
======

Seven is a Python module for Sparse Low-Rank Deconvolution Analysis (SLRDA) of
the fMRI signal (BOLD). It allows to estimate the Haemodynamic Response Function
(HRF) in each voxels of the brain along a with a decomposition of the neural
activity.


Important links
===============

- Official source code repo: https://github.com/CherkaouiHamza/seven

Dependencies
============

The required dependencies to use the software are:

* Numba
* Joblib
* Numpy
* Scipy
* Matplotlib
* Nibabel
* Nilearn
* Sklearn
* Threadpoolctl
* Prox_tv'

License
=======
All material is Free Software: BSD license (3 clause).


Installation
============

In order to perform the installation, run the following command from the seven directory::

    python3 setup.py install --user

To run all the tests, run the following command from the seven directory::

    pytest

Development
===========

Code
----

GIT
~~~

You can check the latest sources with the command::

    git clone git://github.com/CherkaouiHamza/seven

or if you have write privileges::

    git clone git@github.com:CherkaouiHamza/seven
