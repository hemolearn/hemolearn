.. -*- mode: rst -*-

|Python27|_ |Python35|_


.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/scikit-learn


Seven
======

Seven is a Python module for low-rank sparse deconvolution of the fMRI signal (BOLD).


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
* Matplotlib (for examples)


License
=======
All material is Free Software: BSD license (3 clause).


Installation
============

In order to perform the installation, run the following command from the pybold directory::

    python setup.py install --user

To run all the tests, run the following command from the pybold directory::

    python -m unittest discover seven/tests

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
