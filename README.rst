.. -*- mode: rst -*-

|Python36|_ |Travis|_ |Codecov|_


.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
.. _Python36: https://badge.fury.io/py/scikit-learn

.. |Travis| image:: https://travis-ci.com/hcherkaoui/hemolearn.svg?branch=master
.. _Travis: https://travis-ci.com/hcherkaoui/hemolearn


.. |Codecov| image:: https://codecov.io/gh/hcherkaoui/hemolearn/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/hcherkaoui/hemolearn


.. image:: https://i.ibb.co/71j06FR/hemolearn-logo.png


HemoLearn
=========

HemoLearn is a Python module to estimate the Haemodynamic Response Function (HRF)
in brain from resting-state or task fMRI data (BOLD signal). It relies on a
Sparse Low-Rank Deconvolution Analysis (SLRDA) to distangles the
neurovascular coupling from the the neural activity.


Important links
===============

- Official source code repo: https://github.com/hcherkaoui/hemolearn

Dependencies
============

The required dependencies to use the software are:

* Numba >= 0.51.2
* Joblib >= 0.11
* Numpy >= 1.14.0
* Scipy >= 1.0.0
* Matplotlib >= 2.1.2
* Nilearn >= 0.5.2
* Sklearn >= 0.22.1
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

    git clone git://github.com/hcherkaoui/hemolearn

or if you have write privileges::

    git clone git@github.com:hcherkaoui/hemolearn
