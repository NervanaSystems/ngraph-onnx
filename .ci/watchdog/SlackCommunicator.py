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

from slackclient import SlackClient

class SlackCommunicator:
    """Class wrapping SlackClient API

    The purpose of this class is to wrap methods from SlackClient API used in Watchdog, for less error-prone and
    more convenient use. Docs for used API, including wrapped methods can be found at:
    https://slackapi.github.io/python-slackclient/

        :param slack_token:       Token used for Slack
        :type slack_server:       String
    """

    def __init__(self, slack_token):
        self.channel = "ngraph-onnx-ci-alerts"
        self.thread_id = None
        self.queued_messages = []
        self.slack_client = None
        self.slack_token = slack_token

    def queue_message(self, message):
        """Queues message to be sent later.
            
            :param message:     Message content
            :type message:      String
        """
        self.queued_messages.append(message)

    def send_message(self, message):
        """Sends queued messages as single communication.
            
            :param message:     Final message's content
            :type message:      String
        """
        if self.slack_client is None:
           try:
               self.slack_client = SlackClient(self.slack_token)
           except:
               print("!!CRITICAL!! SlackCommunicator::CRITICAL: Could not create client")
               raise
        try:
            final_message = message + "\n\n" + "\n".join(self.queued_messages)
            self.slack_client.api_call(
                "chat.postMessage",
                link_names=1,
                as_user=False,
                username="CI_WATCHDOG",
                channel=self.channel,
                text=final_message,
                thread_ts=self.thread_id)
        except:
            print("!!CRITICAL!! SlackCommunicator: Could not send message")
            raise
