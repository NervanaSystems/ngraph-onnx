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
import os
import json

# Logging
log = logging.getLogger(__name__)
ch = logging.StreamHandler()
log.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
log.addHandler(ch)

# Watchdog static constant variables
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_BUILD_DURATION_THRESHOLD = datetime.timedelta(minutes=60)
_CI_START_THRESHOLD = datetime.timedelta(minutes=10)
_AWAITING_JENKINS_THRESHOLD = datetime.timedelta(minutes=5)
_WATCHDOG_DIR = os.path.expanduser('~')
_PR_REPORTS_CONFIG_KEY = 'pr_reports'
_CI_BUILD_FAIL_MESSAGE = 'ERROR:   py3: commands failed'
_CI_BUILD_SUCCESS_MESSAGE = 'py3: commands succeeded'


class Watchdog:
    """Class describing nGraph-ONNX-CI Watchdog.

    Watchdog connects to GitHub and retrieves the list of current pull requests (PRs) in
    NervanaSystems/ngraph-onnx repository. Then it connects to specified Jenkins server to
    check CI jobs associated with every PR. Watchdog verifies time durations for Jenkins
    initial response, job queue and execution against time treshold constants. Every fail
    is logged and reported through Slack App on channel **ngraph-onnx-ci-alerts**.

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

    def __init__(self, jenkins_token, jenkins_server, jenkins_user, git_token, git_org,
                 git_project, slack_token, ci_job_name, watchdog_job_name):
        self._config_path = os.path.join(_WATCHDOG_DIR, '{}/.{}_ci_watchdog.json'.format(_WATCHDOG_DIR, git_project))
        # Jenkins Wrapper object for CI job
        self._jenkins = JenkinsWrapper(jenkins_token,
                                       jenkins_user=jenkins_user,
                                       jenkins_server=jenkins_server)
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

    def run(self, quiet=False):
        """Run main watchdog logic. 
        
        Retrieve list of pull requests and pass it to the method responsible for checking them.

            :param quiet:   Flag for disabling sending report through Slack
            :type quiet:    Boolean
        """
        try:
            pull_requests = self._git.get_pull_requests()
        except Exception:
            message = 'Failed to retrieve Pull Requests!'
            log.exception(message)
            self._queue_message(message, message_severity='internal')
        try:
            current_prs = []
            # Check all pull requests
            for pr in pull_requests:
                # Append PRs checked in current run for Watchdog config cleanup
                current_prs.append(str(pr.number))
                self._check_pr(pr)
                pr_timestamp = time.mktime(pr.updated_at.timetuple())
                self._config[_PR_REPORTS_CONFIG_KEY][str(pr.number)] = pr_timestamp
            self._update_config(current_prs)
        except Exception as e:
            log.exception(str(e))
            self._queue_message(str(e), message_severity='internal')
        self._send_message(quiet=quiet)

    def _read_config_file(self):
        """Read Watchdog config file stored on the system.

        The file stores every fail already reported along with timestamp. This
        mechanism is used to prevent Watchdog from reporting same failure
        multiple times. In case there's no config under the expected path,
        appropriate data structure is created and returned.

            :return:            Returns dict of dicts with reported fails with their timestamps
            :rtype:             dict of dicts
        """
        if os.path.isfile(self._config_path):
            log.info('Reading config file in: {}'.format(self._config_path))
            file = open(self._config_path, 'r')
            data = json.load(file)
        else:
            log.info('No config file found in: {}'.format(self._config_path))
            data = {_PR_REPORTS_CONFIG_KEY: {}}
        return data

    def _should_ignore(self, pr):
        """
        Determine if PR should be ignored.

            :param pr:          Single PR being currently checked
            :type pr:           github.PullRequest.PullRequest

            :return:            Returns True if PR should be ignored
            :rtype:             Bool
        """
        pr_number = str(pr.number)
        # Ignore PR if base ref is not master
        if 'master' not in pr.base.ref:
            log.info('PR#{} should be ignored. Base ref is not master'.format(pr_number))
            return True
        
        # Ignore PR if mergeable state is 'dirty' or 'behind'.
        # Practically this ignores PR in case of merge conflicts
        ignored_mergeable_states = ['behind', 'dirty']
        for state in ignored_mergeable_states:
            if state in pr.mergeable_state:
                log.info('PR#{} should be ignored. Mergeable state is {} '.format(pr_number, state))
                return True
        
        # Ignore if PR was already checked and there was no update in meantime
        pr_timestamp = time.mktime(pr.updated_at.timetuple())
        if pr_number in self._config[_PR_REPORTS_CONFIG_KEY] and pr_timestamp == \
                self._config[_PR_REPORTS_CONFIG_KEY][pr_number]:
            log.info('PR#{} should be ignored. No update since last check'.format(pr_number))
            return True

        # If no criteria for ignoring PR are met - return false
        return False

    def _check_pr(self, pr):
        """
        Check pull request (if there's no reason to skip). Retrieve list of statuses for every PR's last 
        commit and interpret them.

        Filters out statuses unrelated to nGraph-ONNX Jenkins CI and passes relevant statuses to method
        that interprets them. If no commit statuses related to Jenkins are available after time defined
        by **_AWAITING_JENKINS_THRESHOLD** calls appropriate method to check for builds waiting in queue.

            :param pr:       GitHub Pull Requests
            :type pr:        github.PullRequest.PullRequest
        """
        log.info('===============================================')
        pr_number = str(pr.number)
        if self._should_ignore(pr):
            log.info('Ignoring PR#%s', pr_number)
            return
        log.info('Checking PR#%s', pr_number)
        
        # Find last commit in PR
        last_commit = pr.get_commits().reversed[0]
        
        # Calculate time passed since PR update (any commit, merge or comment)
        pr_time_delta = self._now_time - pr.updated_at
        
        # Get statuses and filter them to contain only those related to Jenkins CI
        # and check if CI in Jenkins started
        statuses = last_commit.get_statuses()
        jenk_statuses = [stat for stat in statuses if
                            'nGraph-ONNX Jenkins CI (IGK)' in stat.context]
        
        # If there's no status after assumed time - check if build is waiting in queue
        if not jenk_statuses:
            log.info('CI for PR %s: NO JENKINS STATUS YET', pr_number)
            if pr_time_delta > _AWAITING_JENKINS_THRESHOLD:
                self._check_missing_status(pr, pr_time_delta)
        else:
            # Interpret found CI statuses
            self._interpret_statuses(jenk_statuses, pr)

    def _check_missing_status(self, pr, pr_time_delta):
        """
        Check if Jenkins build corresponding PR was scheduled.

        This method is used in case no status for nGraph-ONNX CI is present on GitHub.
        Jenkins job corresponding to PR is being searched for CI build. If build is scheduled and waits
        in a queue this is expected behaviour. A warning will be raised if time waiting for available 
        executor exceeds treshold. If no appropriate build is present, it's already executing or
        build does not wait in queue - error is communicated. This means Jenkins did not succesfully 
        pass status to GitHub.

            :param pr:                  Single PR being currently checked
            :param pr_time_delta:       Time since last PR update
            :type pr:                   github.PullRequest.PullRequest
            :type pr_time_delta:        datetime.timedelta
        """
        pr_number = str(pr.number)
        project_name_full = self._ci_job_name + '/PR-' + pr_number

        try:
            # Retrieve console output from last Jenkins build for job corresponding to this PR
            last_build = self._jenkins.get_job_info(project_name_full)['lastBuild']['number']
            console_output = self._jenkins.get_build_console_output(project_name_full, last_build)
        except Exception:
            message = ('PR# {}: missing status on GitHub after {} minutes. '
                    'Jenkins job corresponding to this PR not created!'.format(pr_number, pr_time_delta.seconds / 60))
            self._queue_message(message, message_severity='error', pr=pr)
            return
        # Check if CI build was scheduled - commit hash on GH must match hash in last Jenkins build console output
        # Retrieve hash from Jenkins output
        match_string = '(?:Obtained .ci/[a-zA-Z/]+Jenkinsfile from ([a-z0-9]{40}))'
        match_obj = re.search(match_string, console_output)
        try:
            retrieved_commit_hash = match_obj.group(1)
        except Exception:
            message = ('PR# {}: missing status on GitHub after {} minutes. '
                'Failed to retrieve commit SHA from Jenkins console output!'.format(pr_number, pr_time_delta.seconds / 60))
            self._queue_message(message, message_severity='error', pr=pr)
            return
        # If hash strings don't match then build for that PR's last commit hasn't started yet
        if retrieved_commit_hash != pr.get_commits().reversed[0].sha:
            message = ('PR# {}: missing status on GitHub after {} minutes. '
                    'Jenkins build corresponding to this commit not found!'.format(pr_number, pr_time_delta.seconds / 60))
            self._queue_message(message, message_severity='error', pr=pr)
            return

        # If hash strings match - check if build started executing on machine
        # If it did - Jenkins failed to send status to GitHub
        if 'Running on' in console_output:
            message = ('PR# {}: missing status on GitHub after {} minutes. '
                    'Jenkins build corresponding to this PR is running!'.format(pr_number, pr_time_delta.seconds / 60))
            self._queue_message(message, message_severity='error', pr=pr)
            return

        # If no fail has been detected at this point - status is probably missing due to build waiting in queue
        # Check if build is waiting in queue
        if 'Waiting for next available executor on' in console_output:
            log.info('CI for PR %s: WAITING IN QUEUE', pr_number)
            if  pr_time_delta > _CI_START_THRESHOLD:
                # Log warning if build waits in queue for too long
                message = ('Jenkins CI build for PR# {} still waiting in queue after {}' \
                              ' minutes!'.format(pr_number, pr_time_delta.seconds / 60))
                self._queue_message(message, message_severity='warning', pr=pr)
            return
        
        # If no reason was found for missing status log fail 
        message = 'PR# {}: missing status on GitHub after {} minutes. '.format(pr_number, pr_time_delta.seconds / 60)
        self._queue_message(message, message_severity='error', pr=pr)

    def _interpret_statuses(self, jenk_statuses, pr):
        """
        Loop through commit statuses and validates them.

        This method verifies every commit status for given PR, checking if related Jenkins CI job
        build started within designated time threshold, finished within designated threshold and
        with correct output.

            :param jenk_statuses:       Paginated list of commit statuses filtered out to contain
                                        only Jenkins statuses
            :param pr:                  Single PR being currently checked
            :type jenk_statuses:        github.PaginatedList.PaginatedList of
                                        github.CommitStatus.CommitStatus
            :type pr:                   github.PullRequest.PullRequest
        """
        pr_number = str(pr.number)
        for stat in jenk_statuses:
            try:
                # Retrieve build number for Jenkins build related to this PR
                build_number = self._retrieve_build_number(stat.target_url)
                # CI build finished - verify if expected output is present
                finished_statuses = ['Build finished', 'This commit cannot be built', 'This commit looks good']
                pending_statuses = ['This commit is being built', 'Testing in progress']
                if any(phrase in stat.description for phrase in finished_statuses):
                    self._check_finished(pr, build_number)
                    break
                # CI build in progress - verify timeouts for build queue and duration
                elif any(phrase in stat.description for phrase in pending_statuses):
                    self._check_in_progress(pr, build_number)
                    break
                # CI waiting to start for too long
                elif 'Awaiting Jenkins' in stat.description:
                    self._check_awaiting(pr, build_number, stat.updated_at)
                    break
            except Exception:
                # Log Watchdog internal error in case any status can't be properly verified
                message = 'Failed to verify status "' + stat.description + '" for PR ' + pr_number
                log.exception(message)
                self._queue_message(message, message_severity='internal', pr=pr)
                break

    def _retrieve_build_number(self, url):
        """
        Retrieve Jenkins CI job build number from URL address coming from GitHub commit status.

            :param url:         URL address from GitHub commit status
            :type url:          String

            :return:            Returns build number
            :rtype:             int
        """
        # Retrieve the build number from url string
        match_obj = re.search('(?:/PR-[0-9]+/)([0-9]+)', url)
        try:
            number = int(match_obj.group(1))
            return number
        except Exception:
            log.exception('Failed to retrieve build number from url link: %s', url)
            raise

    def _queue_message(self, message, message_severity, pr = None):
        """
        Add a message to message queue in Slack App object.

        The queued message is constructed based on message string passed as
        a method argument and message header.

        Message header is mapped to message severity also passed as an argument.

            :param message:                 Message content
            :param message_severity:        Message severity level
            :type message:                  String
            :type message_severity:         int
        """
        log.info(message)
        if 'internal' in message_severity:
            message_header = '!!! --- !!! INTERNAL WATCHDOG ERROR !!! --- !!!'
        elif 'error' in message_severity:
            message_header = '!!! nGraph-ONNX CI Error !!!'
        elif 'warning' in message_severity:
            message_header = 'nGraph-ONNX CI WARNING'
        else:
            message_header = 'nGraph-ONNX CI INFO'
        # If message is related to PR attatch url
        if pr:
            message = message + '\n' + pr.html_url
            
        send = message_header + '\n' + message
        self._slack_app.queue_message(send)

    def _check_finished(self, pr, build_number):
        """
        Verify if finished build output contains expected string for either fail or success.

            :param pr:                  Single PR being currently checked
            :param build_number:        Jenkins CI job build number
            :type pr:                   github.PullRequest.PullRequest
            :type build_number:         int
        """
        pr_number = str(pr.number)
        log.info('CI for PR %s: FINISHED', pr_number)
        # Check if FINISH was valid FAIL / SUCCESS
        project_name_full = self._ci_job_name + '/PR-' + pr_number
        build_output = self._jenkins.get_build_console_output(project_name_full, build_number)
        if _CI_BUILD_FAIL_MESSAGE not in build_output \
                and _CI_BUILD_SUCCESS_MESSAGE not in build_output:
            message = ('ONNX CI job for PR #{} finished but no tests success or fail '
                       'confirmation is present in console output!'.format(pr_number))
            self._queue_message(message, message_severity='error', pr=pr)

    def _send_message(self, quiet=False):
        """Send messages queued in Slack App object to designated Slack channel.

            :param quiet:   Flag for disabling sending report through Slack
            :type quiet:    Boolean

        Queued messages are being sent as a single communication.
        """
        if len(self._slack_app.queued_messages) > 0:
            try:
                watchdog_build = self._jenkins.get_job_info(self._watchdog_job_name)['lastBuild']
                watchdog_build_number = watchdog_build['number']
                watchdog_build_link = watchdog_build['url']
            except Exception:
                watchdog_build_number = 'UNKNOWN'
                watchdog_build_link = self._jenkins.jenkins_server
            send = self._watchdog_job_name + '- build ' + str(
                watchdog_build_number) + ' - ' + watchdog_build_link
            self._slack_app.send_message(send, quiet=quiet)
        else:
            log.info('Nothing to report.')

    def _check_in_progress(self, pr, build_number):
        """Check if CI build succesfully started.

        Checks if build started within designated time threshold, and job is
        currently running - it didn't cross the time threshold.

            :param pr:                  Single PR being currently checked
            :param build_number:        Jenkins CI job build number
            :type pr:                   github.PullRequest.PullRequest
            :type build_number:         int
        """
        pr_number = str(pr.number)
        log.info('CI for PR %s: TESTING IN PROGRESS', pr_number)
        project_name_full = self._ci_job_name + '/PR-' + pr_number
        build_info = self._jenkins.get_build_info(project_name_full, build_number)
        build_datetime = datetime.datetime.fromtimestamp(build_info['timestamp'] / 1000.0)
        # If build still waiting in queue
        queue_item = self._jenkins.get_queue_item(build_info['queueId'])
        build_delta = self._now_time - build_datetime
        log.info('Build %s: IN PROGRESS, started: %s minutes ago', str(build_number),
                 str(build_delta))
        # 'why' present if job is in queue and doesnt have executor yet
        if 'why' in queue_item:
            if build_delta > _CI_START_THRESHOLD:
                message = 'ONNX CI job build #{}, for PR #{} waiting in queue after {} ' \
                          'minutes'.format(build_number, pr_number, str(build_delta.seconds / 60))
                self._queue_message(message, message_severity='warning', pr=pr)
                if self.jenkins.get_idle_ci_hosts() > 0:
                    message = 'ONNX CI job build #{}, for PR #{} waiting ' \
                              'in queue, despite idle executors!'.format(build_number, pr_number)
                    self._queue_message(message, message_severity='error', pr=pr)
                return
        if build_delta > _BUILD_DURATION_THRESHOLD:
            # CI job take too long, possibly froze - communicate failure
            message = ('ONNX CI job build #{}, for PR #{} started,'
                       'but did not finish in designated time of {} '
                       'minutes!'.format(build_number, pr_number,
                                         str(_BUILD_DURATION_THRESHOLD.seconds / 60)))
            self._queue_message(message, message_severity='error', pr=pr)

    def _check_awaiting(self, pr, build_number, status_updated_at):
        """
        Check if CI build doesn't take too long to start.

            :param pr:                  Single PR being currently checked
            :param build_number:        Jenkins CI job build number
            :param status_updated_at:   GitHub status update time
            :type pr:                   github.PullRequest.PullRequest
            :type build_number:         int
            :type status_updated_at:    datetime.datetime
        """
        # Calculate time passed since last status update
        delta = self._now_time - status_updated_at
        log.info('CI for PR %s: AWAITING JENKINS', pr.number)
        if delta > _CI_START_THRESHOLD:
            message = 'nGraph-ONNX CI job for PR #{} still awaiting Jenkins after {}' \
                ' minutes!'.format(pr.number, str(delta.seconds / 60))
            self._queue_message(message, message_severity='error', pr=pr)

    def _update_config(self, current_prs):
        """
        Update Watchdog config file with PRs checked in current Watchdog run, remove old entries.

            :param current_prs:        List of PR numbers checked during current Watchdog run
            :type current_prs:         list of ints
        """
        # Cleanup config of old reports
        for pr in self._config[_PR_REPORTS_CONFIG_KEY].copy().keys():
            if pr not in current_prs:
                self._config[_PR_REPORTS_CONFIG_KEY].pop(pr)
        log.info('Writing to config file at: {}'.format(self._config_path))
        file = open(self._config_path, 'w+')
        json.dump(self._config, file)
