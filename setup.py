#!/usr/bin/env python

from os import path
from glob import glob
from distutils.core import setup
from setuptools import find_packages

SOURCES_ROOT = path.abspath(path.dirname(__file__))

setup(name='ngraph-onnx',
      version='0.15.0',
      description='nGraph Backend for ONNX',
      long_description=open(path.join(SOURCES_ROOT, 'README.md')).read(),
      long_description_content_type='text/markdown',
      author='Intel',
      author_email='intelnervana@intel.com',
      url='https://github.com/NervanaSystems/ngraph-onnx',
      license='License :: OSI Approved :: Apache Software License',
      packages=find_packages(exclude=['tests', 'tests.*']),
      data_files=[('', ['LICENSE']), ('licenses', glob(path.join(SOURCES_ROOT, 'licenses/*')))],
      install_requires=['cachetools', 'ngraph-core', 'numpy', 'onnx', 'setuptools'])
