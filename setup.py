#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages
import os

ONNX_IMPORT_SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))

data_files = [
    (
        'licenses',
        [
            os.path.join(ONNX_IMPORT_SOURCE_DIR, 'LICENSE'),
        ],
    ),
]

setup(name='ngraph-onnx',
      version='0.10.0-rc.1',
      description='ONNX support for ngraph',
      author='Intel',
      author_email='intelnervana@intel.com',
      url='http://www.intelnervana.com',
      license='License :: OSI Approved :: Apache Software License',
      packages=find_packages(exclude=['tests', 'tests.*', 'tests_core', 'tests_core.*']),
      data_files=data_files,
      install_requires=['cachetools', 'numpy', 'onnx', 'setuptools'])
