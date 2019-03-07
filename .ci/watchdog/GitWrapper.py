#!/usr/bin/python3

# INTEL CONFIDENTIAL
# Copyright 2018-2019 Intel Corporation All Rights Reserved.
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

import logging
import timeout_decorator
from datetime import datetime
from retrying import retry
from github import Github, GithubException

# Logging
log = logging.getLogger(__name__)
ch = logging.StreamHandler()
log.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
log.addHandler(ch)

_RETRY_LIMIT = 3
_RETRY_COOLDOWN_MS = 2000
_REQUEST_TIMEOUT_S = 10


class GitWrapper:
    """Class wrapping PyGithub API.

    The purpose of this class is to wrap methods from PyGithub API used in Watchdog, for less error-prone and
    more convenient use. Docs for used API, including wrapped methods can be found at:
    https://pygithub.readthedocs.io/en/latest/introduction.html

        :param git_token:       Token used for GitHub
        :param repository:      GitHub repository name
        :param project:         GitHub project name
        :type git_token:        String
        :type repository:       String
        :type project:          String
    """

    def __init__(self, git_token, repository, project):
        self.git = Github(git_token)
        self.repository = repository
        self.project = project

    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_git_time(self):
        """Retrieve time from GitHub.

        Used to reliably determine time during Watchdog run.

            :return:                    Datetime object describing current time
            :rtype:                     datetime
        """
        try:
            datetime_object = self._get_git_time()
        except ValueError as e:
            raise GitWrapperError(str(e))
        except GithubException as e:
            message = 'GitHub Exception during API status retrieval. Exception: {}'.format(str(e))
            raise GitWrapperError(message)
        except timeout_decorator.TimeoutError:
            message = 'GitHub Exception during API status retrieval. Timeout during API request.'
            raise GitWrapperError(message)
        return datetime_object

    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_pull_requests(self):
        """Retrieve paginated list of pull requests from GitHub.

            :return:                    Paginated list of Pull Requests in GitHub repo
            :rtype:                     github.PaginatedList.PaginatedList of github.PullRequest.PullRequest
        """
        try:
            prs = self._get_pull_requests()
        except GithubException as e:
            message = 'GitHub Exception during API status retrieval. Exception: {}'.format(str(e))
            raise GitWrapperError(message)
        return prs

    @timeout_decorator.timeout(_REQUEST_TIMEOUT_S)
    def _get_git_time(self):
        """Private method retrieving time from GitHub.

            :return:                    Datetime object describing current time
            :rtype:                     datetime
        """
        datetime_string = self.git.get_api_status().raw_headers.get('date', '')
        datetime_format = '%a, %d %b %Y %H:%M:%S %Z'
        datetime_object = datetime.strptime(datetime_string, datetime_format)
        return datetime_object

    @timeout_decorator.timeout(_REQUEST_TIMEOUT_S)
    def _get_pull_requests(self):
        """Private method retrieving pull requests from GitHub.

            :return:                    Paginated list of Pull Requests in GitHub repo
            :rtype:                     github.PaginatedList.PaginatedList of github.PullRequest.PullRequest
        """
        return self.git.get_organization(self.repository).get_repo(self.project).get_pulls()

class GitWrapperError(Exception):
    """Base class for exceptions raised in GitWrapper.

        :param message                   Explanation of the error
    """

    def __init__(self, message):
        self.message = message
        log.exception(message)
