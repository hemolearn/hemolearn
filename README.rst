.. -*- mode: rst -*-

|Python35|_ |Travis|_ |Codecov|_


.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/scikit-learn

.. |Travis| image:: https://travis-ci.com/CherkaouiHamza/hemolearn.svg?branch=master
.. _Travis: https://travis-ci.com/CherkaouiHamza/hemolearn


.. |Codecov| image:: https://codecov.io/gh/CherkaouiHamza/hemolearn/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/CherkaouiHamza/hemolearn


HemoLearn
=========

HemoLearn is a Python module to estimate the Haemodynamic Response Function (HRF)
in brain from resting-state or task fMRI data (BOLD signal). It relies on a
Sparse Low-Rank Deconvolution Analysis (SLRDA) to distangles the
neurovascular coupling from the the neural activity.


Important links
===============

- Official source code repo: https://github.com/CherkaouiHamza/hemolearn

Dependencies
============

The required dependencies to use the software are:

* Numba >= 0.41.0
* Joblib >= 0.11
* Numpy >= 1.14.0
* Scipy >= 1.0.0
* Matplotlib >= 2.1.2
* Nibabel >= 2.3.0
* Nilearn >= 0.5.2
* Sklearn >= 0.19.2
* Threadpoolctl >= 1.0.0
* Prox_tv

License
=======

All material is Free Software: BSD license (3 clause).

Installation
============

In order to perform the installation, run the following command from the hemolearn directory::

    python3 setup.py install --user

To run all the tests, run the following command from the hemolearn directory::

    pytest

Development
===========

Code
----

GIT
~~~

You can check the latest sources with the command::

    git clone git://github.com/CherkaouiHamza/hemolearn

or if you have write privileges::

    git clone git@github.com:CherkaouiHamza/hemolearn
