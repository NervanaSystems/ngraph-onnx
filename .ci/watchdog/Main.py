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

import argparse
import sys
from Watchdog import Watchdog

DEFAULT_SLACK_TOKEN_FILE = '/home/lab_nerval/tokens/slack_token'
DEFAULT_GITHUB_TOKEN_FILE = '/home/lab_nerval/tokens/github_token'
DEFAULT_GITHUB_ORGANIZATION = 'NervanaSystems'
DEFAULT_GITHUB_PROJECT = 'ngraph-onnx'
DEFAULT_JENKINS_TOKEN_FILE = '/home/lab_nerval/tokens/crackerjack'
DEFAULT_JENKINS_SERVER = 'https://crackerjack.intel.com/'
DEFAULT_JENKINS_USER = 'lab_nerval'
DEFAULT_CI_JOB_NAME = 'onnx/ngraph-onnx-ci'
DEFAULT_WATCHDOG_JOB_NAME = 'onnx/ci_watchdog'


def main(args):
    """
    Read args passed to script, load tokens and run watchdog.

    Keyword arguments:
    :param args:    arguments parsed by argparse ArgumentParser
    :return:        returns status code 0 on successful completion
    """
    jenkins_server = args.jenkins_server.strip()
    jenkins_user = args.jenkins_user.strip()
    jenkins_token = open(args.jenkins_token).read().replace('\n', '').strip()
    slack_token = open(args.slack_token).read().replace('\n', '').strip()
    github_token = open(args.github_token).read().replace('\n', '').strip()
    github_org = args.github_org
    github_project = args.github_project
    ci_job = args.ci_job.strip()
    watchdog_job = args.watchdog_job.strip()

    wd = Watchdog(jenkins_token=jenkins_token,
                  jenkins_server=jenkins_server,
                  jenkins_user=jenkins_user,
                  git_token=github_token,
                  git_org=github_org,
                  git_project=github_project,
                  slack_token=slack_token,
                  ci_job_name=ci_job,
                  watchdog_job_name=watchdog_job)
    wd.run()

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--slack-token', help='Path to Slack user token to communicate messages.',
                        default=DEFAULT_SLACK_TOKEN_FILE, action='store', required=False)

    parser.add_argument('--github-token', help='Path to GitHub user token to access repo.',
                        default=DEFAULT_GITHUB_TOKEN_FILE, action='store', required=False)

    parser.add_argument('--github-org', help='Name of organization on GitHub.',
                        default=DEFAULT_GITHUB_ORGANIZATION, action='store', required=False)

    parser.add_argument('--github-project', help='Name of project on GitHub.',
                        default=DEFAULT_GITHUB_PROJECT, action='store', required=False)

    parser.add_argument('--jenkins-token', help='Path to Jenkins user token to access build info.',
                        default=DEFAULT_JENKINS_TOKEN_FILE, action='store', required=False)

    parser.add_argument('--jenkins-server', help='Jenkins server address.',
                        default=DEFAULT_JENKINS_SERVER, action='store', required=False)

    parser.add_argument('--jenkins-user', help='Jenkins user used to log in.',
                        default=DEFAULT_JENKINS_USER, action='store', required=False)

    parser.add_argument('--ci-job', help='Jenkins CI job name.',
                        default=DEFAULT_CI_JOB_NAME, action='store', required=False)

    parser.add_argument('--watchdog-job', help='Jenkins CI Watchdog job name.',
                        default=DEFAULT_WATCHDOG_JOB_NAME, action='store', required=False)

    args = parser.parse_args()
    sys.exit(main(args))
