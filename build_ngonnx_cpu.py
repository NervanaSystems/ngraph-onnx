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
import multiprocessing
import os
from subprocess import check_call
import glob

def clone_repo(target_name, repo, *args, version='master'):
    pwd = os.getcwd()

    if os.path.exists(target_name) and os.path.isdir(target_name):
        os.chdir(target_name)
        check_call(['git', 'checkout', version])
        check_call(['git', 'pull'])
    else:
        if len(args) > 0:
            check_call(['git', 'clone', *args, repo, target_name])
        else:
            check_call(['git', 'clone', repo, target_name])

        os.chdir(target_name)
        # checkout the specified branch
        check_call(["git", "checkout", version])

    os.chdir(pwd)

def get_wheel_name(whl_name, search_dir):
    pwd = os.getcwd()
    os.chdir(search_dir)
    wheel_files = glob.glob(whl_name +  '*.whl')
    if len(wheel_files) != 1:
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
    os.makedirs(path, exist_ok=True)

    # Run cmake
    os.chdir('build')

    cmake_cmd = ['cmake']
    cmake_cmd.extend(cmake_flags)
    cmake_cmd.extend([src_location])

    print('nGraph CMAKE flags: {}'.format(cmake_cmd))
    check_call(cmake_cmd)
    n_cores = int(multiprocessing.cpu_count() / 2)
    check_call(["make", "-j" + str(n_cores), "install"])

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
    wheel_path = get_wheel_name('ngraph_onnx', os.path.join(src_location, 'dist'))
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
    exec(
        compile(
            open(activate_this_file, "rb").read(), activate_this_file, 'exec'),
        dict(__file__=activate_this_file), dict(__file__=activate_this_file))

def install_wheel(wheel_path):
    check_call(['pip', 'install', '-U', wheel_path])

def main():
    """
    Builds nGraph, and nGraph-ONNX for python 3
    """

    #-------------------------------
    # Recipe
    #-------------------------------

    # Create the build directory
    build_dir = os.path.abspath('build')
    os.makedirs(build_dir, exist_ok=True)
    print('Build location: ' + os.path.abspath(build_dir))
    os.chdir(build_dir)

    # Component versions
    ngraph_version = "master"

    # Download nGraph
    clone_repo(
        "ngraph",
        "https://github.com/NervanaSystems/ngraph.git",
        "--single-branch",
        version=ngraph_version)

    # Now build nGraph
    ngraph_install_dir = os.path.join(build_dir, 'ngraph_dist')
    ngraph_cmake_flags = [
        '-DCMAKE_BUILD_TYPE=RELEASE',
        '-DCMAKE_INSTALL_PREFIX=' + ngraph_install_dir,
        '-DNGRAPH_TOOLS_ENABLE=YES',
        '-DNGRAPH_USE_PREBUILT_LLVM=TRUE',
        '-DNGRAPH_ONNX_IMPORT_ENABLE=TRUE',
        '-DNGRAPH_UNIT_TEST_ENABLE=FALSE',
        '-DNGRAPH_PYTHON_BUILD_ENABLE=ON',
    ]

    print('----- Build nGraph -----')
    build_ngraph('./ngraph', ngraph_cmake_flags)
    pyngraph_whl = build_pyngraph('./ngraph', ngraph_install_dir)
    print('SUCCESS! Generated wheel: {}'.format(pyngraph_whl))

    print('----- Build nGraph-ONNX -----')
    ngonnx_whl = build_ngraph_onnx('./..')
    print('SUCCESS! Generated wheel: {}'.format(ngonnx_whl))

    # Run a quick test
    print('----- Install and load venv -----')
    venv_dir = './ngonnx'
    install_virtual_env(venv_dir)
    load_venv(venv_dir)

    print('----- Installing nGraph wheel -----')
    install_wheel(pyngraph_whl)
    print('----- Installing nGraph-ONNX wheel -----')
    install_wheel(ngonnx_whl)

    print('----- Test nGraph and nGraph-ONNX packages -----')
    # download test data
    if not os.path.exists('./resnet50'):
        check_call(['wget',
                    'https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz'])
        check_call(['tar', '-xzvf', 'resnet50.tar.gz'])

    import onnx
    import ngraph as ng
    from ngraph_onnx.onnx_importer.backend import NgraphBackend
    import numpy as np

    ng.abs([[1, 2, 3], [4, 5, 6]])
    # <Abs: 'Abs_1' ([2, 3])>

    test_data = np.load('./resnet50/test_data_0.npz', encoding='bytes')
    inputs = list(test_data['inputs'])
    model = onnx.load('./resnet50/model.onnx')
    prepared_model = NgraphBackend.prepare(model, 'CPU')
    outputs = list(prepared_model.run(inputs))
    ref_outputs = test_data['outputs']
    np.testing.assert_equal(len(ref_outputs), len(outputs))
    for idx, _ in enumerate(outputs):
        np.testing.assert_equal(ref_outputs[idx].dtype, outputs[idx].dtype)
        np.testing.assert_allclose(ref_outputs[idx], outputs[idx], rtol=1e-3, atol=1e-7)

    print('SUCCESS!')

if __name__ == '__main__':
    main()
