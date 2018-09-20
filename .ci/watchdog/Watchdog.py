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
import argparse
import sys
import os
import json

# Logging
log = logging.getLogger(__file__)
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
_WATCHDOG_JOB_NAME = "Onnx_CI_Watchdog"
_CI_BUILD_FAIL_MESSAGE = "ERROR:   py3: commands failed"
_CI_BUILD_SUCCESS_MESSAGE = "py3: commands succeeded"

class Watchdog:
    def __init__(self, jenkins_token, git_token, slack_token, ci_job_name):
        # Jenkins Wrapper object for CI job
        self.jenk = JenkinsWrapper(jenkins_token)
        # Load GitHub token and log in, retrieve pull requests
        self.git = GitWrapper(git_token, repository='NervanaSystems', project='ngraph-onnx')
        # Create Slack api object
        self.slack_app = SlackCommunicator(slack_token)
        self._ci_job_name = ci_job_name
        # Read config file
        self.config = self._read_config_file()
        # Time at Watchdog initiation
        self.now_time = self.git.get_git_time()

    def run(self):
        try:
            pulls = self.git.pull_requests
        except:
            message = "Failed to retrieve Pull Requests!"
            log.exception(message)
            self._queue_message(message, message_severity=999)
        try:
            self._check_prs(pulls)
        except Exception as e:
            log.exception(str(e))
            self._queue_message(str(e), message_severity=999)
        self._send_message()

    def _check_prs(self, pulls)
        current_prs = []
        # --- MAIN LOGIC ---
        # Check all pull requests
        for pr in pulls:
            log.info("Checking PR#%s", pr.number)
            current_prs.append(str(pr.number))
            last_commit = (pr.get_commits().reversed)[0]
            statuses = last_commit.get_statuses()
            # Filter statuses to contain only those related to Jenkins CI and check if CI in Jenkins started
            jenk_statuses = [stat for stat in statuses if "Jenkins CI" in stat.context]
            delta = self.now_time - pr.updated_at
            if not jenk_statuses and (delta > _AWAITING_JENKINS_TRESHOLD):
                message = "Jenkins CI report for PR# {} not present on GitHub after {} minutes!".format(pr.number, delta.seconds / 60)
                self._queue_fail(message, pr)
            # Interpret CI statuses
            for stat in jenk_statuses:
                try:
                    if "Build finished" in stat.description:
                        log.info("\tCI for PR %s: FINISHED", str(pr.number))
                        # Check if FINISH was valid FAIL / SUCCESS
                        build_no = self._retrieve_build_number(stat.target_url)
                        build_output = self.get_build_console_output(self._ci_job_name, build_no)
                        if _CI_BUILD_FAIL_MESSAGE not in build_output and _CI_BUILD_SUCCESS_MESSAGE not in build_output:
                            message = ("Onnx CI job for PR #{} finished but "
                                        "no tests success or fail confirmation is present in console output!")
                            self._queue_fail(message, pr)
                        break
                    # CI build in progress                
                    elif "Testing in progress" in stat.description:
                        build_no = self._retrieve_build_number(stat.target_url)
                        self._check_ci_build(pr, build_no)
                        break
                    # CI waiting to start
                    elif "Awaiting Jenkins" in stat.description and (now_time - stat.updated_at > _CI_START_TRESHOLD):
                        # CI job failed to start for given amount of time
                        message = "Onnx CI job for PR #{} still awaiting Jenkins after {} minutes!".format(pr.number, str(now_time - stat.updated_at))
                        self._queue_fail(message, pr)
                        break
                except:
                    message = "\tFailed to verify status \"" + stat.description + "\" for PR " + str(pr.number)
                    log.exception(message)
                    self._queue_message(message, message_severity=999)
                    break
        self._update_config(current_prs)

    def _read_config_file(self):
        if os.path.isfile(_CONFIG_PATH):
            log.info("Config file exists, reading.")
            file = open(_CONFIG_PATH,'r')
            data = json.load(file)
        else:
            log.info("No config file.")
            data = { _PR_REPORTS_CONFIG_KEY: {} }
        return data

    def _retrieve_build_number(self, url):
        job_info = self.jenk.get_job_info(self._ci_job_name)
        oldest_build = job_info['builds'][-1]['number']
        # Retrieve the build number
        matchObj = re.search("(?:/" + job + "/)([0-9]+)",url)
        try:
            number = int(matchObj.group(1))
            if number < oldest_build:
                log.exception("Build number: %s doesnt exist, the oldest build is: %s", str(number), str(oldest_build))
                raise
            return number
        except:
            log.exception("Failed to retrieve build number from url link: %s", url)
            raise

    def _queue_message(self, message, message_severity):
        log.info(message)
        if message_severity == 999:
            message_header = "!!! --- !!! INTERNAL WATCHDOG ERROR !!! --- !!!"
        if message_severity == 3:
            message_header = "!!! nGraph-ONNX CI Error !!!"
        elif message_severity == 2:
            message_header = "nGraph-ONNX CI WARNING"
        else:
            message_header = "nGraph-ONNX CI INFO"
        send = message_header + "\n" + message
        self.slack_app.queue_message(send,severity=message_severity)

    def _queue_fail(self, message, pr, message_severity=3):
        pr_timestamp = time.mktime(pr.updated_at.timetuple())
        if str(pr.number) not in self.config[_PR_REPORTS_CONFIG_KEY] or pr_timestamp > self.config[_PR_REPORTS_CONFIG_KEY][str(pr.number)]:
            self.config[_PR_REPORTS_CONFIG_KEY][str(pr.number)] = pr_timestamp
            send = message + "\n" + pr.html_url
            self._queue_message(send, message_severity)
        else:
            log.info("PR " + str(pr.number) + " -- fail already reported.")

    def _send_message(self):
        if len(self.slack_app.queued_messages) > 0:
            watchdog_build = self.jenk.get_job_info(_WATCHDOG_JOB_NAME)['lastBuild']
            watchdog_build_number = watchdog_build['number']
            watchdog_build_link = watchdog_build['url']
            send = "nGraph-ONNX CI Watchdog - build " + str(watchdog_build_number) + " - " + watchdog_build_link
            self.slack_app.send_message(send, final=True, severity=0)
        else:
            log.info("Nothing to report.")

    def _check_ci_build(self, pr, build_number):
        build_info = self.jenk.get_build_info(self._ci_job_name, build_number)
        build_datetime = datetime.datetime.fromtimestamp(build_info['timestamp']/1000.0)
        # If build still waiting in queue
        try:
            queueItem = self.jenk.get_queue_item(build_info['queueId'])
            # 'why' present if job is in queue and doesnt have executor yet
            if "why" in queueItem:
                delta = self.now_time - build_datetime
                if delta > _CI_START_TRESHOLD:
                    message = "Onnx CI job build #{}, for PR #{} waiting in queue after {} minutes".format(build_number, pr.number, str(delta.seconds / 60))
                    self._queue_fail(message, pr, message_severity=2)
                    if get_idle_ci_hosts() > 0:
                        message = "Onnx CI job build #{}, for PR #{} waiting in queue, despite idle executors!".format(build_number, pr.number)
                        self._queue_fail(message, pr)
                    return
        except:
            # Failure to get queue item means build has started on a node
            pass
        log.info("\tBuild %s: IN PROGRESS, started: %s", str(build_number), str(build_datetime))
        delta = self.now_time - build_datetime
        if delta > _BUILD_DURATION_TRESHOLD:
            # CI job take too long, possibly froze - communiate failure
            message = "Onnx CI job build #{}, for PR #{} started, but did not finish in designated time of {} minutes!".format(build_number, pr.number, str(_BUILD_DURATION_TRESHOLD))
            self.queue_fail(message, pr)        

    # Write config data structure to file
    def _update_config(self, current_prs):
        # Cleanup config of old reports
        for pr in self.config[_PR_REPORTS_CONFIG_KEY].copy().keys():
            if pr not in current_prs:
                self.config[_PR_REPORTS_CONFIG_KEY].pop(pr)
        log.info("Writting to config file.")
        file = open(_CONFIG_PATH, "w+")
        json.dump(self.config, file)
