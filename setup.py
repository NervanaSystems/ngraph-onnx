#!/usr/bin/env python

from glob import glob
from distutils.core import setup
from setuptools import find_packages

setup(name='ngraph-onnx',
      version='0.10.0',
      description='nGraph Backend for ONNX',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      author='Intel',
      author_email='intelnervana@intel.com',
      url='https://github.com/NervanaSystems/ngraph-onnx',
      license='License :: OSI Approved :: Apache Software License',
      packages=find_packages(exclude=['tests', 'tests.*', 'tests_core', 'tests_core.*']),
      data_files=[('', ['LICENSE']), ('licenses', glob('licenses/*'))],
      install_requires=['cachetools', 'ngraph-core', 'numpy', 'onnx', 'setuptools'])
