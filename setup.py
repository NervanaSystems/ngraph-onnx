#!/usr/bin/env python

from distutils.core import setup

from setuptools import find_packages

setup(name='ngraph-onnx',
      version='0.10.0-rc.1',
      description='ONNX support for ngraph',
      author='Intel',
      author_email='intelnervana@intel.com',
      url='http://www.intelnervana.com',
      license='License :: OSI Approved :: Apache Software License',
      packages=find_packages(exclude=['tests', 'tests_core']),
      install_requires=['cachetools', 'ngraph-core', 'numpy', 'onnx', 'setuptools'])
