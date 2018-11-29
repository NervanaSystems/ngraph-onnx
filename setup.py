#!/usr/bin/env python

from glob import glob
from distutils.core import setup
from setuptools import find_packages

setup(name='ngraph-onnx',
      version='0.10.0-rc.5',
      description='ONNX support for ngraph',
      author='Intel',
      author_email='intelnervana@intel.com',
      url='http://www.intelnervana.com',
      license='License :: OSI Approved :: Apache Software License',
      packages=find_packages(exclude=['tests', 'tests.*', 'tests_core', 'tests_core.*']),
      data_files=[('', ['LICENSE']), ('licenses', glob('licenses/*'))],
      install_requires=['cachetools', 'numpy', 'onnx', 'setuptools'])
