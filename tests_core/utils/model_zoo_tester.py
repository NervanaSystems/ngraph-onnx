# ******************************************************************************
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

from collections import defaultdict
import glob
import os
import shutil
import tarfile
import tempfile

from six.moves.urllib.request import urlretrieve

import onnx.backend.test
from onnx.backend.test.case.test_case import TestCase as OnnxTestCase


class ModelZooTestRunner(onnx.backend.test.BackendTest):

    def __init__(self, backend, models_dict, parent_module=None):
        # type: (Type[Backend], Dict[str,str], Optional[str]) -> None
        self.backend = backend
        self._parent_module = parent_module
        self._include_patterns = set()  # type: Set[Pattern[Text]]
        self._exclude_patterns = set()  # type: Set[Pattern[Text]]
        self._test_items = defaultdict(dict)  # type: Dict[Text, Dict[Text, TestItem]]

        for model_name, url in models_dict.items():
            test_name = "test_{}".format(model_name)

            test_case = OnnxTestCase(
                name=test_name,
                url=url,
                model_name=model_name,
                model_dir=None,
                model=None,
                data_sets=None,
                kind='OnnxBackendRealModelTest',
            )
            self._add_model_test(test_case, 'Zoo')

    def _prepare_model_data(self, model_test):  # type: (TestCase) -> Text
        onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', os.path.join('~', '.onnx')))
        models_dir = os.getenv('ONNX_MODELS', os.path.join(onnx_home, 'models'))
        model_dir = os.path.join(models_dir, model_test.model_name)  # type: Text

        # If model already exists, exit
        if os.path.exists(os.path.join(model_dir, 'model.onnx')):
            return model_dir

        # If model does not exist, but directory does exist, move directory
        if os.path.exists(model_dir):
            bi = 0
            while True:
                dest = '{}.old.{}'.format(model_dir, bi)
                if os.path.exists(dest):
                    bi += 1
                    continue
                shutil.move(model_dir, dest)
                break

        # Download and extract model and data
        download_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            download_file.close()
            print('Start downloading model {} from {}'.format(
                model_test.model_name, model_test.url))
            urlretrieve(model_test.url, download_file.name)
            print('Done')

            with tempfile.TemporaryDirectory() as temp_extract_dir:
                with tarfile.open(download_file.name) as tar_file:
                    tar_file.extractall(temp_extract_dir)

                # Move expected files from temp_extract_dir to temp_clean_dir
                temp_clean_dir = tempfile.mkdtemp()

                model_files = glob.glob(temp_extract_dir + '/**/*.onnx', recursive=True)
                assert len(model_files) > 0, 'Model file not found for {}'.format(model_test.name)
                model_file = model_files[0]
                shutil.move(model_file, temp_clean_dir + '/model.onnx')
                for test_data_set in glob.glob(temp_extract_dir + '/**/test_data_set_*',
                                               recursive=True):
                    shutil.move(test_data_set, temp_clean_dir)
                for test_data_set in glob.glob(temp_extract_dir + '/**/test_data_*.npz',
                                               recursive=True):
                    shutil.move(test_data_set, temp_clean_dir)

                # Move temp_clean_dir to final destination
                shutil.move(temp_clean_dir, model_dir)

        except Exception as e:
            print('Failed to prepare data for model {}: {}'.format(
                model_test.model_name, e))
            raise
        finally:
            os.remove(download_file.name)
        return model_dir
