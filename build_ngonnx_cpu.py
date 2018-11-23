#!/usr/bin/env python3
# ==============================================================================
#  Copyright 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import argparse
import errno
import os
from subprocess import check_call
import sys
import glob

def clone_repo(target_name, repo, *args, version='master'):
    # First git clone
    if len(args) > 0:
        check_call(["git", "clone", *args, repo, target_name])
    else:
        check_call(["git", "clone", repo, target_name])

    # Next goto this folder nd determine the name of the root folder
    pwd = os.getcwd()
    # Go to the tree
    os.chdir(target_name)

    # checkout the specified branch
    check_call(["git", "checkout", version])
    os.chdir(pwd)

def get_wheel_name(whl_name, search_dir):
    pwd = os.getcwd()
    os.chdir(search_dir)
    wheel_files = glob.glob(whl_name +  '*.whl')
    if (len(wheel_files) != 1):
        raise Exception('Error getting the ' + whl_name + ' wheel file')

    output_wheel = wheel_files[0]
    print( "OUTPUT WHL FILE: {}".format(output_wheel))

    os.chdir(pwd)
    return output_wheel

def build_ngraph(src_location, cmake_flags):
    pwd = os.getcwd()

    src_location = os.path.abspath(src_location)
    print('Source location: ' + src_location)

    os.chdir(src_location)

    # mkdir build directory
    path = 'build'
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass

    # Run cmake
    os.chdir('build')

    cmake_cmd = ['cmake']
    cmake_cmd.extend(cmake_flags)
    cmake_cmd.extend([src_location])

    print('nGraph CMAKE flags: {}'.format(cmake_cmd))
    result = call(cmake_cmd)
    if (result != 0):
        raise Exception('Error running command: ' + str(cmake_cmd))

    result = call(["make", "-j $(lscpu --parse=CORE | grep -v '#' | sort | uniq | wc -l)", "install"])
    if (result != 0):
        raise Exception('Error running command: make -j install')
    os.chdir(pwd)

def build_pyngraph(src_location, ngraph_install_dir):
    pwd = os.getcwd()

    src_location = os.path.abspath(src_location)
    print('Source location: ' + src_location)

    pyngraph_dir = os.path.join(src_location, 'python')
    os.chdir(pyngraph_dir)

    clone_repo('pybind11', 'https://github.com/pybind/pybind11.git', '--recursive')

    os.environ['PYBIND_HEADERS_PATH'] = os.path.join(pyngraph_dir, 'pybind11')
    os.environ['NGRAPH_CPP_BUILD_PATH'] = ngraph_install_dir
    os.environ['NGRAPH_ONNX_IMPORT_ENABLE'] = 'TRUE'

    check_call(['python3', 'setup.py', 'bdist_wheel'])
    wheel_path = get_wheel_name('ngraph', os.path.join(pyngraph_dir, 'dist'))
    wheel_path = os.path.join(pyngraph_dir, 'dist', wheel_path)
    os.chdir(pwd)
    return wheel_path

def build_ngraph_onnx(src_location):
    pwd = os.getcwd()

    src_location = os.path.abspath(src_location)
    print('Source location: ' + src_location)

    os.chdir(src_location)
    check_call(['python3', 'setup.py', 'bdist_wheel'])
    wheel_path = get_wheel_name('ngraph-onnx', os.path.join(src_location, 'dist'))
    wheel_path = os.path.join(src_location, 'dist', wheel_path)

    os.chdir(pwd)
    return wheel_path

def install_virtual_env(venv_dir):
    # Check if we have virtual environment
    # TODO

    # Setup virtual environment
    venv_dir = os.path.abspath(venv_dir)
    # Note: We assume that we are using Python 3 (as this script is also being
    # executed under Python 3 as marked in line 1)
    check_call(["virtualenv", "-p", "/usr/bin/python3", venv_dir])

def load_venv(venv_dir):
    activate_this_file = os.path.abspath(venv_dir) + "/bin/activate_this.py"
    # The execfile API is for Python 2. We keep here just in case you are on an
    # obscure system without Python 3
    # execfile(activate_this_file, dict(__file__=activate_this_file))
    exec(
        compile(
            open(activate_this_file, "rb").read(), activate_this_file, 'exec'),
        dict(__file__=activate_this_file), dict(__file__=activate_this_file))

def main():
    '''
    Builds TensorFlow, ngraph, and ngraph-tf for python 3
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--debug_build',
        help="Builds a debug version of the nGraph components\n",
        action="store_true")

    arguments = parser.parse_args()

    if (arguments.debug_build):
        print("Building in DEBUG mode\n")

    #-------------------------------
    # Recipe
    #-------------------------------

if __name__ == '__main__':
    main()
