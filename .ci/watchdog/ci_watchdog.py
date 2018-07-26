#!/usr/bin/python2.7

# INTEL CONFIDENTIAL
# Copyright 2017 Intel Corporation All Rights Reserved.
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

from github import Github
import jenkins
import datetime
import re
import logging
from SlackCommunicator import SlackCommunicator
import argparse
import sys

log = logging.getLogger(__file__)
ch = logging.StreamHandler()
log.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
log.addHandler(ch)

slack_token_file="/home/lab_nerval/tokens/slack_token"

github_token_file="/home/lab_nerval/tokens/github_token"

jenkins_server="http://10.91.54.11:8080/"
jenkins_token_file = "/home/lab_nerval/tokens/scheduler"
jenkins_user = 'lab_nerval'

def communicate_fail(message, pr_link, slack_app):
    log.info("[DEBUG] [INFO] %s", message)
    slack_app.send_message("!!! Onnx_CI CRITICAL FAILURE !!!\n" + message + "\n" + pr_link, final=True, severity=3)

def build_output(jenkins, build_number, job):
    try:
        output = jenkins.get_build_console_output(job,build_number)
    except:
        log.exception("Failed to retrieve console output for build: %s", str(build_number))
        output = ""
    return output

def retrieve_build_number(url,job):
    # Retrieve the build number
        matchObj = re.search("(?:/" + job + "/)([0-9]+)",url)
        try:
            number = int(matchObj.group(1))
            return number
        except:
            log.exception("Failed to retrieve build number from url link: %s", url)
            return -1

def main(args):
    # Reading passed args
    slack_token = open(args.slack_token).read().replace('\n','').strip()
    github_token = open(args.github_token).read().replace('\n','').strip()
    jenkins_user = args.jenkins_user
    jenkins_token = open(args.jenkins_token).read().replace('\n','').strip()

    # Default variables
    job_name = 'Onnx_CI'
    valid_footer = "___________________________________ summary ____________________________________"
    build_duration_treshold = datetime.timedelta(minutes=40)
    ci_start_treshold = datetime.timedelta(minutes=40)
    now_time = datetime.datetime.now()

    # Create Slack api object
    slack_app = SlackCommunicator(slack_token)

    # Load github token and log in, retrieve pull requests
    git = Github(github_token)
    pulls = git.get_organization('NervanaSystems').get_repo('ngraph-onnx').get_pulls()

    # Load jenkins token and log in, retrieve job list
    jenk = jenkins.Jenkins(jenkins_server,username=jenkins_user,password=jenkins_token)
    ci_job = jenk.get_job_info(job_name)

    # Check all pull requests
    for pr in pulls:
        log.info("Checking PR#%s", pr.number)
        last_commit = (pr.get_commits().reversed)[0]
        statuses = last_commit.get_statuses()
        # Filter statuses to contain only those related to Jenkins CI and check if CI in Jenkins started
        jenk_statuses = [stat for stat in statuses if "Jenkins CI" in stat.context]
        other_statuses = [stat for stat in statuses if not "Jenkins CI" in stat.context]
        if not jenk_statuses:
            log.info("\tJenkins CI for PR#%s not scheduled yet.", pr.number)
        for stat in jenk_statuses:
            # If CI build finished
            try:
                if "Build finished" in stat.description:
                    build_no = retrieve_build_number(stat.target_url, job_name)
                    log.info("\tBuild %s: FINISHED", str(build_no))
                    if valid_footer not in build_output(jenk,build_no, job_name):
                        communicate_fail("Onnx CI job build #{}, for PR #{} failed critically!".format(build_no, pr.number), pr.html_url, slack_app)
                    else:
                        log.info("\tCI build %s for PR #%s finished successfully.", str(build_no), str(pr.number))
                    break
                # CI build in progress                
                elif "Testing in progress" in stat.description:
                    build_no = retrieve_build_number(stat.target_url, job_name)
                    build_timestamp = (jenk.get_build_info(job_name,build_no))['timestamp']
                    build_start_time = datetime.datetime.fromtimestamp(build_timestamp/1000.0)
                    log.info("\tBuild %s: IN PROGRESS, started: %s", str(build_no), str(build_start_time))
                    if now_time - build_start_time > build_duration_treshold:
                        # CI job froze, communiate failure
                        communicate_fail("Onnx CI job build #{}, for PR #{} started, but did not finish in designated time!".format(build_no, pr.number), pr.html_url, slack_app)
                    break
                # CI waiting to start
                elif "Awaiting Jenkins" in stat.description and (now_time - stat.updated_at > ci_start_treshold):
                    # CI job failed to start for given amount of time
                    communicate_fail("Onnx CI job for PR #{} failed to start in designated time!".format(pr.number), pr.html_url, slack_app)
                    break
            except:
                log.exception("\tFailed to verify status \"%s\" for PR#%s", stat.description, str(pr.number))
        # for stat in other_statuses:
        #     if "error" in stat.state:
        #         communicate_fail("Onnx CI failed for PR #{}. Failed context: {}".format(pr.number, stat.context), pr.html_url, slack_app)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--slack-token", help="Path to token for Slack user to communicate messages.",
                        default=slack_token_file, action="store", required=False)
    
    parser.add_argument("--github-token", help="Path to token for GitHub user to access repo.",
                        default=github_token_file, action="store", required=False)
    
    parser.add_argument("--jenkins-user", help="Jenkins username for login.",
                        default=jenkins_user, action="store", required=False)
    
    parser.add_argument("--jenkins-token", help="Path to token for Jenkins user to access build info.",
                        default=jenkins_token_file, action="store", required=False)

    args = parser.parse_args()
    sys.exit(main(args))