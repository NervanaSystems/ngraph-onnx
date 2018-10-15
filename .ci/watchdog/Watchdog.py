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

import datetime
import time
import re
import logging
from SlackCommunicator import SlackCommunicator
from JenkinsWrapper import JenkinsWrapper
from GitWrapper import GitWrapper
import argparse
import sys
import os
import json

# Logging
log = logging.getLogger(__name__)
ch = logging.StreamHandler()
log.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
log.addHandler(ch)

# Watchdog static constant variables
_CONFIG_PATH = "/tmp/onnx_ci_watchdog.json"
_BUILD_DURATION_TRESHOLD = datetime.timedelta(minutes=60)
_CI_START_TRESHOLD = datetime.timedelta(minutes=10)
_AWAITING_JENKINS_TRESHOLD = datetime.timedelta(minutes=5)
_PR_REPORTS_CONFIG_KEY = "pr_reports"
_CI_BUILD_FAIL_MESSAGE = "ERROR:   py3: commands failed"
_CI_BUILD_SUCCESS_MESSAGE = "py3: commands succeeded"

class Watchdog:
    """Class describing nGraph-ONNX-CI Watchdog.

    Watchdog connects to GitHub and retrieves the list of current pull requests (later abbreviated PRs) 
    in NervanaSystems/ngraph-onnx repository. Then it connects to specified Jenkins server to check CI jobs 
    associated with every PR. Watchdog verifies time durations for Jenkins initial response, job queue and 
    execution against time treshold constants. Every fail is logged and reported through Slack App on
    channel **ngraph-onnx-ci-alerts**.

        :param jenkins_token:       Token used for Jenkins
        :param jenkins_server:      Jenkins server address
        :param jenkins_user:        Username used to connect to Jenkins
        :param git_token:           Token used to connect to GitHub
        :param slack_token:         Token used to connect to Slack App
        :param ci_job_name:         nGraph-ONNX CI job name used in Jenkins
        :param watchdog_job_name:   Watchdog job name used in Jenkins
        :type jenkins_token:        String
        :type jenkins_server:       String
        :type jenkins_user:         String
        :type git_token:            String
        :type slack_token:          String
        :type ci_job_name:          String
        :type watchdog_job_name:    String

        .. note::
            Watchdog and nGraph-ONNX CI job must be placed on the same Jenkins server.
    """
    
    def __init__(self, jenkins_token, jenkins_server, jenkins_user, git_token, git_org, git_project, slack_token, ci_job_name, watchdog_job_name):
        # Jenkins Wrapper object for CI job
        self._jenkins = JenkinsWrapper(jenkins_token, jenkins_user=jenkins_user, jenkins_server=jenkins_server)
        # Load GitHub token and log in, retrieve pull requests
        self._git = GitWrapper(git_token, repository=git_org, project=git_project)
        # Create Slack api object
        self._slack_app = SlackCommunicator(slack_token)
        self._ci_job_name = ci_job_name
        self._watchdog_job_name = watchdog_job_name
        # Read config file
        self._config = self._read_config_file()
        # Time at Watchdog initiation
        self._now_time = self._git.get_git_time()

    def run(self):
        """Runs watchdog main logic. 
        
        Retrievs list of pull requests and passes it to the method responsible for checking them.
        """
        try:
            pull_requests = self._git.get_pull_requests()
        except:
            message = "Failed to retrieve Pull Requests!"
            log.exception(message)
            self._queue_message(message, message_severity=999)
        try:
            self._check_prs(pull_requests)
        except Exception as e:
            log.exception(str(e))
            self._queue_message(str(e), message_severity=999)
        self._send_message()

    def _read_config_file(self):
        """Reads Watchdog config file stored on the system. 
        
        The file stores every fail already reported along with 
        timestamp. This mechanism is used to prevent Watchdog from reporting same failure multiple times. In case 
        there's no config under the expected path, appropriate data structure is created and returned.

            :return:            Returns dict of dicts with reported fails with their timestamps
            :rtype:             dict of dicts
        """
        
        if os.path.isfile(_CONFIG_PATH):
            log.info("Config file exists, reading.")
            file = open(_CONFIG_PATH,'r')
            data = json.load(file)
        else:
            log.info("No config file.")
            data = { _PR_REPORTS_CONFIG_KEY: {} }
        return data

    def _check_prs(self, pull_requests):
        """ 
        Loops through pull requests, retrieving list of statuses for every PR's last commit. Filters out statuses 
        unrelated to nGraph-ONNX Jenkins CI and passes relevant statuses to method that interprets them. If no 
        commit statuses related to Jenkins are availible after time defined by**_AWAITING_JENKINS_TRESHOLD** reports
        fail.
        This method also updates Watchdog config with current Pull Requests.

            :param pull_requests:       Paginated list of Pull Requests
            :type pull_requests:        github.PaginatedList.PaginatedList of github.PullRequest.PullRequest
        """
        
        current_prs = []
        # Check all pull requests
        for pr in pull_requests:
            log.info("===============================================")
            pr_number = str(pr.number)
            log.info("Checking PR#%s", pr_number)
            # Append PRs checked in current run for Watchdog config cleanup
            current_prs.append(pr_number)
            # Find last commit in PR
            last_commit = (pr.get_commits().reversed)[0]
            # Calculate time passed since PR update (any commit, merge or comment)
            pr_delta = self._now_time - pr.updated_at
            # Get statuses and filter them to contain only those related to Jenkins CI and check if CI in Jenkins started
            statuses = last_commit.get_statuses()
            jenk_statuses = [stat for stat in statuses if "nGraph-ONNX Jenkins CI (IGK)" in stat.context]
            # Fail if there are no statuses related to Jenkins after assumed time
            if not jenk_statuses and (pr_delta > _AWAITING_JENKINS_TRESHOLD):
                message = "Jenkins CI report for PR# {} not present on GitHub after {} minutes!".format(pr_number, pr_delta.seconds / 60)
                self._queue_fail(message, pr)
            else:
                # Interpret found CI statuses
                self._interpret_statuses(jenk_statuses, pr)
        self._update_config(current_prs)

    def _interpret_statuses(self, jenk_statuses, pr):
        """Loops through commit statuses and validates them. 
        
        This method verifies every commit status for given PR, checking if related Jenkins CI job build started within 
        designated time treshold, finished within designated treshold and with correct output.

            :param jenk_statuses:       Paginated list of commit statuses filtered out to contain only Jenkins statuses
            :param pr:                  Single PR being currently checked
            :type jenk_statuses:        github.PaginatedList.PaginatedList of github.CommitStatus.CommitStatus
            :type pr:                   github.PullRequest.PullRequest
        """

        pr_number = str(pr.number)
        for stat in jenk_statuses:
            try:
                # Retrieve build number for Jenkins build related to this PR
                build_number = self._retrieve_build_number(stat.target_url)
                # Calculate time passed since last status update
                stat_delta = self._now_time - stat.updated_at
                # CI build finished - verify if expected output is present
                if "Build finished" in stat.description:
                    self._verify_build_output(pr, build_number)
                    break
                # CI build in progress - verify timeouts for build queue and duration
                elif "Testing in progress" in stat.description:
                    self._check_ci_build(pr, build_number)
                    break
                # CI waiting to start for too long
                elif "Awaiting Jenkins" in stat.description:
                    log.info("CI for PR %s: AWAITING JENKINS", pr_number)
                    if stat_delta > _CI_START_TRESHOLD:
                        message = "Onnx CI job for PR #{} still awaiting Jenkins after {} minutes!".format(pr_number, str(stat_delta.seconds / 60))
                        self._queue_fail(message, pr)
                    break
            except:
                # Log Watchdog internal error in case any status can't be properly verified
                message = "Failed to verify status \"" + stat.description + "\" for PR " + pr_number
                log.exception(message)
                self._queue_fail(message, pr, message_severity=999)
                break

    def _retrieve_build_number(self, url):
        """Retrieves Jenkins CI job build number from URL address comming from GitHub commit status.

            :param url:         URL address from GitHub commit status
            :type url:          String

            :return:            Returns build number
            :rtype:             int
        """

        # Get oldest build number
        job_info = self._jenkins.get_job_info(self._ci_job_name)
        # Retrieve the build number from url string
        matchObj = re.search("(?:/" + self._ci_job_name.split("/")[-1] + "/)([0-9]+)",url)
        try:
            number = int(matchObj.group(1))
            return number
        except:
            log.exception("Failed to retrieve build number from url link: %s", url)
            raise

    def _queue_message(self, message, message_severity):
        """Adds message to message queue in Slack App object.

        The queued message is constructed based on message string passed as a method argument and message header.
        Nessage header is mapped to message severity also passed as an argument.

            :param message:                 Message content
            :param message_severity:        Message severity level
            :type message:                  String
            :type message_severity:         int
        """

        log.info(message)
        if message_severity is 999:
            message_header = "!!! --- !!! INTERNAL WATCHDOG ERROR !!! --- !!!"
        elif message_severity is 3:
            message_header = "!!! nGraph-ONNX CI Error !!!"
        elif message_severity is 2:
            message_header = "nGraph-ONNX CI WARNING"
        else:
            message_header = "nGraph-ONNX CI INFO"
        send = message_header + "\n" + message
        self._slack_app.queue_message(send)

    def _verify_build_output(self, pr, build_number):
        """Verifies if finished build output contains expected string for either fail or success.

            :param pr:                  Single PR being currently checked
            :param build_number:        Jenkins CI job build number
            :type pr:                   github.PullRequest.PullRequest
            :type build_number:         int
        """
        pr_number = str(pr.number)
        log.info("CI for PR %s: FINISHED", pr_number)
        # Check if FINISH was valid FAIL / SUCCESS
        build_output = self._jenkins.get_build_console_output(self._ci_job_name, build_number)
        if _CI_BUILD_FAIL_MESSAGE not in build_output and _CI_BUILD_SUCCESS_MESSAGE not in build_output:
            message = ("Onnx CI job for PR #{} finished but "
                "no tests success or fail confirmation is present in console output!".format(pr_number))
            self._queue_fail(message, pr)

    def _queue_fail(self, message, pr, message_severity=3):
        """Adds message to message queue with content and message_severity passed as arguments. The fail message is 
        only being queued if it wasn't already reported.

            :param message:                 Fail message content
            :param pr:                      Single PR being currently checked, related to fail message
            :param message_severity:        Message severity level
            :type message:                  String
            :type pr:                       github.PullRequest.PullRequest
            :type message_severity:         int
        """
        pr_number = str(pr.number)
        pr_timestamp = time.mktime(pr.updated_at.timetuple())
        if pr_number not in self._config[_PR_REPORTS_CONFIG_KEY] or pr_timestamp > self._config[_PR_REPORTS_CONFIG_KEY][pr_number]:
            self._config[_PR_REPORTS_CONFIG_KEY][pr_number] = pr_timestamp
            send = message + "\n" + pr.html_url
            self._queue_message(send, message_severity)
        else:
            log.info("PR " + pr_number + " -- fail already reported.")

    def _send_message(self):
        """Sends messages queued in Slack App object to designated Slack channel. Queued messages are being sent as a single
        communication.
        """
        
        if len(self._slack_app.queued_messages) > 0:
            try:
                watchdog_build = self._jenkins.get_job_info(self._watchdog_job_name)['lastBuild']
                watchdog_build_number = watchdog_build['number']
                watchdog_build_link = watchdog_build['url']
            except:
                watchdog_build_number = "UNKNOWN"
                watchdog_build_link = self._jenkins.jenkins_server
            send = self._watchdog_job_name + "- build " + str(watchdog_build_number) + " - " + watchdog_build_link
            self._slack_app.send_message(send)
        else:
            log.info("Nothing to report.")

    def _check_ci_build(self, pr, build_number):
        """Checks if CI build succesfully started within designated time treshold, and if job is currently running - 
        if it didn't cross the time treshold.

            :param pr:                  Single PR being currently checked
            :param build_number:        Jenkins CI job build number
            :type pr:                   github.PullRequest.PullRequest
            :type build_number:         int
        """
        pr_number = str(pr.number)
        log.info("CI for PR %s: TESTING IN PROGRESS", pr_number)
        build_info = self._jenkins.get_build_info(self._ci_job_name, build_number)
        build_datetime = datetime.datetime.fromtimestamp(build_info['timestamp']/1000.0)
        # If build still waiting in queue
        queueItem = self._jenkins.get_queue_item(build_info['queueId'])
        build_delta = self._now_time - build_datetime
        log.info("Build %s: IN PROGRESS, started: %s minutes ago", str(build_number), str(build_delta))
        # 'why' present if job is in queue and doesnt have executor yet
        if "why" in queueItem:
            if build_delta > _CI_START_TRESHOLD:
                message = "Onnx CI job build #{}, for PR #{} waiting in queue after {} minutes".format(build_number, pr_number, str(build_delta.seconds / 60))
                self._queue_fail(message, pr, message_severity=2)
                if self.jenkins.get_idle_ci_hosts() > 0:
                    message = "Onnx CI job build #{}, for PR #{} waiting in queue, despite idle executors!".format(build_number, pr_number)
                    self._queue_fail(message, pr)
                return
        if build_delta > _BUILD_DURATION_TRESHOLD:
            # CI job take too long, possibly froze - communiate failure
            message = ("Onnx CI job build #{}, for PR #{} started," 
                        "but did not finish in designated time of {} minutes!".format(build_number, pr_number, str(_BUILD_DURATION_TRESHOLD.seconds / 60)))
            self._queue_fail(message, pr)        

    # Write config data structure to file
    def _update_config(self, current_prs):
        """Updates Watchdog config file with PRs checked in current Watchdog run, removes old entries.

            :param current_prs:        List of PR numbers checked during current Watchdog run
            :type current_prs:         list of ints
        """
        
        # Cleanup config of old reports
        for pr in self._config[_PR_REPORTS_CONFIG_KEY].copy().keys():
            if pr not in current_prs:
                self._config[_PR_REPORTS_CONFIG_KEY].pop(pr)
        log.info("Writting to config file.")
        file = open(_CONFIG_PATH, "w+")
        json.dump(self._config, file)
