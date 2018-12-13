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
from jenkins import NotFoundException
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
        self._current_prs = {}
        # Check all pull requests
        for pr in pull_requests:
            try:
                self._check_pr(pr)
            except Exception as e:
                log.exception(str(e))
                self._queue_message(str(e), message_severity='internal')
        self._update_config()
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

    def _check_pr(self, pr):
        """Check pull request (if there's no reason to skip).

        Retrieve list of statuses for every PR's last commit and interpret them. Filters out statuses
        unrelated to nGraph-ONNX Jenkins CI and passes relevant statuses to method that interprets them.
        If no commit statuses related to Jenkins are available after time defined by
        **_AWAITING_JENKINS_THRESHOLD** calls appropriate method to check for builds waiting in queue.

        :param pr:       GitHub Pull Requests
        :type pr:        github.PullRequest.PullRequest
        """
        log.info('===============================================')
        log.info('Checking PR#{}'.format(pr.number))
        # Get last Jenkins status
        last_status = self._get_last_status(pr)
        # Append PR checked in current run for Watchdog config
        self._current_prs[str(pr.number)] = self._get_pr_timestamps(pr, last_status)
        if self._should_ignore(pr) or self._updated_since_last_run(pr):
            log.info('Ignoring PR#{}'.format(pr.number))
            return

        log.info('Last status: {} at {}'.format(last_status.description, last_status.updated_at))
        # Calculate time passed since PR update (any commit, merge or comment)
        pr_time_delta = self._now_time - pr.updated_at
        # If there's no status after assumed time - check if build is waiting in queue
        if pr_time_delta > _CI_START_THRESHOLD and not last_status:
            log.info('CI for PR {}: NO JENKINS STATUS YET'.format(pr.number))
            self._check_missing_status(pr)
        else:
            # Interpret found CI statuses
            self._interpret_status(last_status, pr)

    @staticmethod
    def _get_pr_timestamps(pr, last_status):
        """Get dict containing PR timestamp and last status timestamp.

        :param pr:          Single PR being currently checked
        :type pr:           github.PullRequest.PullRequest

        :return:            Dictionary with PR and last status update timestamps
        :rtype:             dict
        """
        pr_timestamp = time.mktime(pr.updated_at.timetuple())
        if last_status:
            status_timestamp = time.mktime(last_status.updated_at.timetuple())
        else:
            status_timestamp = None
        pr_dict = {'pr_timestamp': pr_timestamp,
                   'status_timestamp': status_timestamp}
        return pr_dict

    @staticmethod
    def _get_last_status(pr):
        """Get last commit status posted from Jenkins.

        :param pr:          Single PR being currently checked
        :type pr:           github.PullRequest.PullRequest

        :return:            Either last PR status posted from Jenkins or None
        :rtype:             github.CommitStatus.CommitStatus
        """
        # Find last commit in PR
        last_commit = pr.get_commits().reversed[0]
        # Get statuses and filter them to contain only those related to Jenkins CI
        # and check if CI in Jenkins started
        statuses = last_commit.get_statuses()
        jenk_statuses = [stat for stat in statuses if
                         'nGraph-ONNX Jenkins CI (IGK)' in stat.context]
        try:
            last_status = jenk_statuses[0]
        except IndexError:
            last_status = None
        return last_status

    @staticmethod
    def _should_ignore(pr):
        """Determine if PR should be ignored.

        :param pr:          Single PR being currently checked
        :type pr:           github.PullRequest.PullRequest

        :return:            Returns True if PR should be ignored
        :rtype:             Bool
        """
        # Ignore PR if base ref is not master
        if 'master' not in pr.base.ref:
            log.info('PR#{} should be ignored. Base ref is not master'.format(pr.number))
            return True

        # Ignore PR if mergeable state is 'dirty' or 'behind'.
        # Practically this ignores PR in case of merge conflicts
        ignored_mergeable_states = ['behind', 'dirty']
        for state in ignored_mergeable_states:
            if state in pr.mergeable_state:
                log.info('PR#{} should be ignored. Mergeable state is {} '.format(pr.number, state))
                return True

        # If no criteria for ignoring PR are met - return false
        return False

    def _updated_since_last_run(self, pr):
        # Ignore if PR was already checked and there was no update in meantime
        pr_number = str(pr.number)
        current_pr_timestamps = self._current_prs.get(pr_number)
        last_pr_timestamps = self._config[_PR_REPORTS_CONFIG_KEY].get(pr_number)
        if current_pr_timestamps == last_pr_timestamps:
            log.info('PR#{} - No update since last check'.format(pr.number))
            return True
        else:
            return False

    def _check_missing_status(self, pr):
        """Verify if missing status is expected.

        This method checks if CI build for last was scheduled and still waits in queue for
        executor.

        :param pr:                  Single PR being currently checked
        :type pr:                   github.PullRequest.PullRequest
        """
        pr_time_delta = self._now_time - pr.updated_at
        build_number = self._build_scheduled(pr)
        if self._build_in_queue(pr, build_number):
            message = ('PR# {}: build waiting in queue after {} minutes.'
                       .format(pr.number, pr_time_delta.seconds / 60))
            severity = 'warning'
        else:
            message = ('PR# {}: missing status on GitHub after {} minutes.'
                       .format(pr.number, pr_time_delta.seconds / 60))
            severity = 'error'
        self._queue_message(message, message_severity=severity, pr=pr)

    def _build_scheduled(self, pr):
        """Check if Jenkins build corresponding to PR was scheduled.

        This method takes last Jenkins build for given PR and compares hash from Jenkins console output
        and sha from PR object to determine if CI build for appropriate commit was scheduled.

        :param pr:                  Single PR being currently checked
        :type pr:                   github.PullRequest.PullRequest

        :return:            Returns build number or -1 if no build found
        :rtype:             int
        """
        pr_number = str(pr.number)
        project_name_full = self._ci_job_name + '/PR-' + pr_number

        try:
            # Retrieve console output from last Jenkins build for job corresponding to this PR
            last_build_number = self._jenkins.get_job_info(project_name_full)['lastBuild']['number']
            console_output = self._jenkins.get_build_console_output(project_name_full, last_build_number)
            # Check if CI build was scheduled - commit hash on GH must match hash in last Jenkins build console output
            # Retrieve hash from Jenkins output
            match_string = '(?:Obtained .ci/[a-zA-Z/]+Jenkinsfile from ([a-z0-9]{40}))'
            retrieved_sha = re.search(match_string, console_output).group(1)
            if retrieved_sha == pr.get_commits().reversed[0].sha:
                return last_build_number
            else:
                return -1
        except (NotFoundException, AttributeError):
            message = ('PR #{}: Jenkins build corresponding to commit {} not found!'
                       .format(pr_number, pr.get_commits().reversed[0].sha))
            self._queue_message(message, message_severity='error', pr=pr)
            return -1

    def _build_in_queue(self, pr, build_number):
        """Check if Jenkins build waits in queue.

        This method verifies if CI build is waiting in queue based on console output.

        :param pr:                  Single PR being currently checked
        :param build_number:        Jenkins build number to retrieve console output from
        :type pr:                   github.PullRequest.PullRequest
        :type build_number:         int

        :return:            Returns True if CI build is waiting in queue
        :rtype:             Bool
        """
        pr_number = str(pr.number)
        project_name_full = self._ci_job_name + '/PR-' + pr_number
        # Retrieve console output
        try:
            console_output = self._jenkins.get_build_console_output(project_name_full, build_number)
        except NotFoundException:
            return False
        # Check if build is waiting in queue (and not already running on an executor)
        if 'Waiting for next available executor on' in console_output \
                and 'Running on' not in console_output:
            log.info('CI for PR %s: WAITING IN QUEUE', pr_number)
            return True
        else:
            return False

    def _interpret_status(self, status, pr):
        """
        Loop through commit statuses and validates them.

        This method verifies every commit status for given PR, checking if related Jenkins CI job
        build started within designated time threshold, finished within designated threshold and
        with correct output.

        :param status:       Paginated list of commit statuses filtered out to contain
                                    only Jenkins statuses
        :param pr:                  Single PR being currently checked
        :type status:        github.PaginatedList.PaginatedList of
                                    github.CommitStatus.CommitStatus
        :type pr:                   github.PullRequest.PullRequest
        """
        try:
            # Retrieve build number for Jenkins build related to this PR
            build_number = self._retrieve_build_number(status.target_url)
            # CI build finished - verify if expected output is present
            finished_statuses = ['Build finished', 'This commit cannot be built', 'This commit looks good']
            pending_statuses = ['This commit is being built', 'Testing in progress']
            if any(phrase in status.description for phrase in finished_statuses):
                self._check_finished(pr, build_number)
            # CI build in progress - verify timeouts for build queue and duration
            elif any(phrase in status.description for phrase in pending_statuses):
                self._check_in_progress(pr, build_number)
        except Exception:
            # Log Watchdog internal error in case any status can't be properly verified
            message = 'Failed to verify status "{}" for PR# {}'.format(status.descriptio, pr.number)
            log.exception(message)
            self._queue_message(message, message_severity='internal', pr=pr)

    def _retrieve_build_number(self, url):
        """Retrieve Jenkins CI job build number from URL address coming from GitHub commit status.

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

    def _queue_message(self, message, message_severity, pr=None):
        """Add a message to message queue in Slack App object.

        The queued message is constructed based on message string passed as
        a method argument and message header. Message header is mapped to message severity
        also passed as an argument.

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
        """Verify if finished build output contains expected string for either fail or success.

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

        Queued messages are being sent as a single communication.

        :param quiet:   Flag for disabling sending report through Slack
        :type quiet:    Boolean
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
        build_delta = self._now_time - build_datetime
        log.info('Build %s: IN PROGRESS, started: %s minutes ago', str(build_number),
                 str(build_delta))
        # If build still waiting in queue
        if build_delta > _CI_START_THRESHOLD and self._build_in_queue(pr, build_number):
            message = ('ONNX CI job build #{}, for PR #{} waiting in queue after {} '
                       'minutes'.format(build_number, pr_number, str(build_delta.seconds / 60)))
            self._queue_message(message, message_severity='warning', pr=pr)
        elif build_delta > _BUILD_DURATION_THRESHOLD:
            # CI job take too long, possibly froze - communicate failure
            message = ('ONNX CI job build #{}, for PR #{} started,'
                       'but did not finish in designated time of {} '
                       'minutes!'.format(build_number, pr_number,
                                         str(_BUILD_DURATION_THRESHOLD.seconds / 60)))
            self._queue_message(message, message_severity='error', pr=pr)

    def _update_config(self):
        """Update Watchdog config file with PRs checked in current Watchdog run, remove old entries.

        :param current_prs:        List of PR numbers checked during current Watchdog run
        :type current_prs:         list of ints
        """
        # Cleanup config of old reports
        log.info('Writing to config file at: {}'.format(self._config_path))
        new_config = {_PR_REPORTS_CONFIG_KEY: self._current_prs}
        file = open(self._config_path, 'w+')
        json.dump(new_config, file)
