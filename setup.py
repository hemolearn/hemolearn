#! /usr/bin/env python
""" Seven setup routine.
"""
import sys
import os
from setuptools import setup, find_packages
# Author: Hamza Cherkaoui
# License: new BSD


def load_version():
    """Executes seven/info.py in a globals dictionary and return it.

    Note: importing Seven is not an option because there may be
    dependencies like nibabel which are not installed and
    setup.py is supposed to install them.
    """
    # load all vars into globals, otherwise
    #   the later function call using global vars doesn't work.
    globals_dict = {}
    with open(os.path.join('seven', 'info.py')) as fp:
        exec(fp.read(), globals_dict)
    return globals_dict


def is_installing():
    # Allow command-lines such as "python setup.py build install"
    install_commands = set(['install', 'develop'])
    return install_commands.intersection(set(sys.argv))


# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

_VERSION_GLOBALS = load_version()
DISTNAME = 'seven'
DESCRIPTION = __doc__
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Hamza Cherkaoui'
MAINTAINER_EMAIL = 'hamza.cherkaoui@inria.fr'
URL = 'https://github.com/CherkaouiHamza/seven'
LICENSE = 'new BSD'
VERSION = _VERSION_GLOBALS['__version__']


if __name__ == "__main__":
    if is_installing():
        module_check_fn = _VERSION_GLOBALS['_check_module_dependencies']
        module_check_fn(is_pyta_installing=True)

    install_requires = \
            ['{0}>={1}'.format(mod, meta['min_version'])
             for mod, meta in _VERSION_GLOBALS['REQUIRED_MODULE_METADATA']
             if meta['required_at_installation']]

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Programming Language :: Python :: 2.7',
          ],
          packages=find_packages(),
          install_requires=install_requires,
          )
