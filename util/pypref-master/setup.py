# -*- coding: utf-8 -*-
"""
Setup file for pypref (Sykline Computation and Database Preferences in Python)

pypref originally used Cython, this setup file just installs the generated C++ code


To install the package, type 

  python setup.py install
  
on the command line within this directory

"""


from distutils.core import setup
from distutils.extension import Extension

#from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize

# C++ Extensions (BNL)

# Read RST file
from os import path
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'readme.rst'), encoding='utf-8') as f:
    long_description = f.read()

# Abhängigkeiten
import numpy

extensions = [
    Extension("bnl", sources=["bnl.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=["-O3"], language="c++")
]

setup(

  name = 'pypref',
    
  version = '0.0.2',
    
  description = long_description.split('\n')[0],
 
  long_description=long_description, 
 
  # make pypref/__init__.py available as "pypref" module
  #packages = find_packages(exclude=['contrib', 'docs', 'tests']),

  # provide mtcars data set as example data
  package_data = { 'pypref': ['mtcars.csv'] },
    
  # Package homepage
  url = 'http://p-roocks.de/rPref/index.php?section=pypref',
  
  # Author details
  author = 'Patrick Roocks',
  author_email = 'mail@p-roocks.de',

  # Choose your license
  license = 'GNU General Public License v3 (GPLv3)',
  
  classifiers=[
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Information Analysis',

    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

    'Programming Language :: Python :: 3.5',
    ],

  keywords='preferences skyline pareto',

  # External BNL algorithm (using numpy)
  ext_modules = cythonize(extensions, language_level = "3"),
  include_dirs = [numpy.get_include()]
)