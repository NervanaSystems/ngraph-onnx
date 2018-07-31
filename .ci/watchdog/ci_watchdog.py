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
import json
from xml.dom.minidom import parseString

log = logging.getLogger(__file__)
ch = logging.StreamHandler()
log.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
log.addHandler(ch)

slack_token_file = "/home/lab_nerval/tokens/slack_token"

github_token_file = "/home/lab_nerval/tokens/github_token"

jenkins_server = "http://10.91.54.11:8080/"
jenkins_token_file = "/home/lab_nerval/tokens/scheduler"
jenkins_user = 'lab_nerval'

ci_host_config="/tmp/onnx_ci_watchdog.json"
# default value for time for updating hosts (in hours)
ci_host_config_update = 24

# Communicate fail through slack only if it hasn't been reported yet
def communicate_fail(message, pr, slack_app, config):
    pr_update = pr.updated_at
    if pr.number not in config['pr_reports'] or pr_update > config['pr_reports'][pr.number]:
        config['pr_reports'][pr.number] = datetime.datetime.now()
        log.info("[DEBUG] [INFO] %s", message)
        slack_app.send_message("!!! Onnx_CI CRITICAL FAILURE !!!\n" + message + "\n" + pr.html_url, final=True, severity=3)
    return config

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

# Get hosts labeled in Jenkins with 'ci' and 'onnx'. Return them in format for storing information in watchdog config file
def get_ci_hosts(jenkins, now_datetime)
    log.info("Reading NGraph-ONNX CI hosts.")
    nodes = jenkins.get_nodes()
    ci_hosts = []
    for node in nodes:
        name = node["name"]
        if name == "master":
            continue
        node_conf = parseString(jenkins.get_node_config(name))
        labels = (node_conf.getElementsByTagName("label")[0].firstChild.wholeText).split(' ')
        if 'ci' in labels and 'onnx' in labels:
            ci_hosts.append(name)
    hosts_dict = {'hosts': ci_hosts,'timestamp': now_datetime}
    return ci_hosts

# Return config structure cleaned of old PRs
def cleanup_prs(config, current_prs):
    for pr in config['pr_reports'].keys():
        if pr not in current_prs:
            config['pr_reports'].pop(pr)
    return config

# Read if config file exists, else create it
def read_config_file(ci_host_config=ci_host_config):
    if os.path.isfile(ci_host_config):
        log.info("Config file exists, reading.")
        file = open(ci_host_config,'r')
        data = json.load(file)
    else:
        log.info("No config file.")
        data = { 'ci_hosts': {'hosts': [],'timestamp': datetime.datetime.fromtimestamp(0)}, 'pr_reports': {} }
    return data

# Write config data structure to file
def update_config(config_dict):
    file = open(ci_host_config,"w+")
    json.dump(config_dict, file)

def main(args):
    # --- PREPARE VARIABLES ---
    # Reading passed args
    slack_token = open(args.slack_token).read().replace('\n','').strip()
    github_token = open(args.github_token).read().replace('\n','').strip()
    jenkins_user = args.jenkins_user
    jenkins_token = open(args.jenkins_token).read().replace('\n','').strip()
    hosts_update = datetime.timedelta(hours=args.hosts_update)
    # Default variables
    job_name = 'Onnx_CI'
    build_duration_treshold = datetime.timedelta(minutes=60)
    ci_start_treshold = datetime.timedelta(minutes=10)
    now_time = datetime.datetime.now()
    # Create Slack api object
    slack_app = SlackCommunicator(slack_token)
    # Load github token and log in, retrieve pull requests
    git = Github(github_token)
    pulls = git.get_organization('NervanaSystems').get_repo('ngraph-onnx').get_pulls()
    # Load jenkins token and log in, retrieve job list
    jenk = jenkins.Jenkins(jenkins_server,username=jenkins_user,password=jenkins_token)
    ci_job = jenk.get_job_info(job_name)
    # Read config file
    config = read_config_file()
    # Update hosts in config if older than 24 hours
    if now_time - config['ci_hosts']['timestamp'] > hosts_update:
        config['ci_hosts'] = get_ci_hosts(jenk, now_time)
    # List of current PR numbers for easier access
    current_prs = []

    # --- MAIN LOGIC ---
    # Check all pull requests
    for pr in pulls:
        current_prs.append(pr.number)
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
                    console_output = jenk.get_build_console_output(job_name, build_no)
                    #if "FAILURE" in (jenk.get_build_info(job_name, build_no)["result"]):
                    if "test session starts" not in build_output(jenk, build_no, job_name):
                        #config = communicate_fail("Onnx CI job build #{}, for PR #{}, failed to run tests!".format(build_no, pr.number), pr, slack_app)
                        log.info("\tCI build %s for PR #%s finished with failure.", str(build_no), str(pr.number))
                    else:
                        log.info("\tCI build %s for PR #%s finished successfully.", str(build_no), str(pr.number))
                    break
                # CI build in progress                
                elif "Testing in progress" in stat.description:
                    build_no = retrieve_build_number(stat.target_url, job_name)
                    build_info = jenk.get_build_info(job_name, build_no)
                    # If build finished in Jenkins but is in progress in GitHub
                    if build_info['result']:
                        config = communicate_fail("Onnx CI job build #{}, for PR #{} finished, but failed to inform GitHub of its results!".format(build_no, pr.number), pr, slack_app)
                        break
                    build_datetime = datetime.datetime.fromtimestamp(build_info['timestamp']/1000.0)
                    # If build still waiting in queue
                    queueId = build_info['queueId']
                    queueItem = jenk.get_queue_item(queueId)
                    try:
                        if "why" in queueItem and now_time - build_datetime > ci_start_treshold:
                            config = communicate_fail("Onnx CI job build #{}, for PR #{} still waiting in queue!".format(build_no, pr.number), pr, slack_app)
                    except:
                        pass
                    log.info("\tBuild %s: IN PROGRESS, started: %s", str(build_no), str(build_start_time))
                    if now_time - build_datetime > build_duration_treshold:
                        # CI job take too long, possibly froze - communiate failure
                        config = communicate_fail("Onnx CI job build #{}, for PR #{} started, but did not finish in designated time!".format(build_no, pr.number), pr, slack_app)
                    break
                # CI waiting to start
                elif "Awaiting Jenkins" in stat.description and (now_time - stat.updated_at > ci_start_treshold):
                    # CI job failed to start for given amount of time
                    config = communicate_fail("Onnx CI job for PR #{} failed to start in designated time!".format(pr.number), pr, slack_app)
                    break
            except:
                log.exception("\tFailed to verify status \"%s\" for PR#%s", stat.description, str(pr.number))
        # for stat in other_statuses:
        #     if "error" in stat.state:
        #         communicate_fail("Onnx CI failed for PR #{}. Failed context: {}".format(pr.number, stat.context), pr.html_url, slack_app)
    # --- CLEANUP & CONFIG UPDATE ---
    config = cleanup_prs(config, current_prs)
    update_config(config)
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
    
    parser.add_argument("--hosts-update", help="Time in hours host config is going to be updated.",
                        default=ci_host_config_update, action="store", required=False)

    args = parser.parse_args()
    sys.exit(main(args))