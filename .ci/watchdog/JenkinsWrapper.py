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
log = logging.getLogger(__name__)
ch = logging.StreamHandler()
log.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
log.addHandler(ch)

_RETRY_LIMIT = 3
_RETRY_COOLDOWN_MS = 5000

class JenkinsWrapper:
    """Class wrapping Python-Jenkins API.

    The purpose of this class is to wrap methods from Python-Jenkins API used in Watchdog, for less error-prone and
    more convenient use. Docs for used API, including wrapped methods can be found at:
    https://python-jenkins.readthedocs.io/en/latest/

        :param jenkins_token:       Token used for Jenkins
        :param jenkins_user:        Username used to connect to Jenkins
        :param jenkins_server:      Jenkins server address
        :type jenkins_token:        String
        :type jenkins_user:         String
        :type jenkins_server:       String
    """
    
    def __init__(self, jenkins_token, jenkins_user, jenkins_server):
        self.jenkins_server = jenkins_server
        self.jenkins = jenkins.Jenkins(jenkins_server, username=jenkins_user, password=jenkins_token)

    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_build_console_output(self, job_name, build_number):
        return self.jenkins.get_build_console_output(job_name, build_number)

    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_job_info(self, job_name):
        return self.jenkins.get_job_info(job_name)
    
    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_build_info(self, job_name, build_number):
        return self.jenkins.get_build_info(job_name, build_number)

    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_queue_item(self, queueId):
        """Method attempts to retrieve Jenkins job queue item. Exception communicating queue doesn't exist is expected,
        in that case method returns empty dict.

            :param queueId:             Jenkins job queue ID number
            :type jenkins_server:       int
            :return:                    Dictionary representing Jenkins job queue item
            :rtype:                     dict
        """
        try:
            return self.jenkins.get_queue_item(queueId)
        except Exception as e:
            # Exception 'queue does not exist' is expected behaviour when job is running
            if "queue" in str(e) and "does not exist" in str(e):
                return {}
            else:
                raise

    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_idle_ci_hosts(self):
        """Method sends GET request to Jenkins server, querrying for idle servers labeled for nGraph-ONNX CI job.

            :return:                    Number of idle hosts delegated to nGraph-ONNX CI
            :rtype:                     int
        """
        jenkins_request_url = self.jenkins_server + "label/ci&&onnx/api/json?pretty=true"
        try:
            log.info("Sending request to Jenkins: %s", jenkins_request_url)
            r = requests.Request(method='GET',url=jenkins_request_url)
            response = self.jenkins.jenkins_request(r).json()
            return (int(response['totalExecutors']) - int(response['busyExecutors']))
        except Exception as e:
            log.exception("Failed to send request to Jenkins!\nException message: %s",str(e))
            raise
