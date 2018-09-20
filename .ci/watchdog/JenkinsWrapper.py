#!/usr/bin/python3

# INTEL CONFIDENTIAL
# Copyright 2018 Intel Corporation All Rights Reserved.
# The source code contained or described herein and all documents related to the
# source code ("Material") are owned by Intel Corporation or its suppliers or
# licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material may contain trade secrets and proprietary
# and confidential information of Intel Corporation and its suppliers and
# licensors, and is protected by worldwide copyright and trade secret laws and
# treaty provisions. No part of the Material may be used, copied, reproduced,
# modified, published, uploaded, posted, transmitted, distributed, or disclosed
# in any way without Intel's prior express written permission.
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery of
# the Materials, either expressly, by implication, inducement, estoppel or
# otherwise. Any license under such intellectual property rights must be express
# and approved by Intel in writing.
# Include any supplier copyright notices as supplier requires Intel to use.
# Include supplier trademarks or logos as supplier requires Intel to use,
# preceded by an asterisk. An asterisked footnote can be added as follows:
# *Third Party trademarks are the property of their respective owners.
# Unless otherwise agreed by Intel in writing, you may not remove or alter
# this notice or any other notice embedded in Materials by Intel or Intel's
# suppliers or licensors in any way.

from time import sleep
import requests
import jenkins
import logging

# Logging
log = logging.getLogger(__file__)
ch = logging.StreamHandler()
log.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
log.addHandler(ch)

_JENKINS_SERVER = "http://10.91.54.11:8080/"
_JENKINS_USER = "lab_nerval"
_RETRY_LIMIT = 3
_RETRY_COOLDOWN = 15

class JenkinsWrapper:
    def __init__(self, jenkins_token):
        self.jenkins = jenkins.Jenkins(_JENKINS_SERVER, username=_JENKINS_USER, password=jenkins_token)

    def _try_jenkins(self, method, args=[]):
        attempt = 0
        while(attempt < _RETRY_LIMIT):
            try:
                return method(*args)
            except Exception as e:
                # Special case for get_queue_item
                if "queue" in str(e) and "does not exist" in str(e):
                    return {}
                attempt = attempt + 1
                log.warning("Failed to execute " + method.__name__ + " -- attempt: " + str(attempt) + " -- ERROR: " + str(e))
            sleep(_RETRY_COOLDOWN)
        raise RuntimeError("Unable to execute " + method.__name__ + " after " + str(_RETRY_LIMIT) + " retries.")

    def get_build_console_output(self, job_name, build_number):
        return self._try_jenkins(self.jenkins.get_build_console_output,[job_name, build_number])

    def get_job_info(self, job_name):
        return self._try_jenkins(self.jenkins.get_job_info,[job_name])

    def get_build_info(self, job_name, build_number):
        return self._try_jenkins(self.jenkins.get_build_info,[job_name, build_number])

    def get_queue_item(self, queueId):
        return self._try_jenkins(self.jenkins.get_queue_item, [queueId])

    def get_idle_ci_hosts(self):
        jenkins_request_url = _JENKINS_SERVER + "label/ci&&onnx/api/json?pretty=true"
        try:
            log.info("Sending request to Jenkins: %s", jenkins_request_url)
            r = requests.Request(method='GET',url=jenkins_request_url)
            response = self._try_jenkins(self.jenkins.jenkins_request, [r]).json()
            return (response['totalExecutors'] - response['busyExecutors'])
        except Exception as e:
            log.exception("Failed to send request to Jenkins!\nException message: %s",str(e))
            raise
