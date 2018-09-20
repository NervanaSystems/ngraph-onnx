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

import arrow

from slackclient import SlackClient
from multiprocessing.pool import ThreadPool

class SlackCommunicator:
    def __init__(self, slack_token):
        self.channel = "ngraph-onnx-ci-alerts"
        self.thread_id = None
        self.queued_messages = []
        self.slack_client = None
        self.slack_token = slack_token

    def _async_flush(queued_msg):
        self.send_message(message=queued_msg["msg"], severity=queued_msg["severity"])

    def _severity_to_string(severity):
        if severity == 0:
            return "[INFO] "
        elif severity == 1:
            return "<WARN> "
        elif severity == 2:
            return "<ERROR> "
        elif severity == 3:
            return "!!CRITICAL!! "
        else:
            return ""

    def queue_message(self, message, severity=0):
        self.queued_messages.append({"msg" : message, "severity" : severity})

    def send_message(self, message, final=False, severity=0):
        
        
        if self.slack_client is None:
           try:
               self.slack_client = SlackClient(self.slack_token)
           except:
               print("!!CRITICAL!! SlackCommunicator::CRITICAL: Could not create client")
               raise
        try:
            if final:
                response = self.slack_client.api_call(
                   "chat.postMessage",
                   as_user=False,
                   username="CI_WATCHDOG",
                   channel=self.channel,
                   text=message)
                self.thread_id = response['ts']

                pool = ThreadPool(4) 
                for queued_msg in self.queued_messages:
                    pool.apply_async(self._async_flush, [queued_msg])
                pool.close()
                pool.join()

                self.slack_client.api_call(
                   "chat.update",
                   channel=self.channel,
                   as_user=True,
                   ts=self.thread_id,
                   text=message)
            else:
                final_message = self._severity_to_string(severity) + message
                if severity > 1:
                    final_message = "*" + final_message + "*"

                if severity == 0:
                    self.thread_infos += 1
                elif severity == 1:
                    self.thread_warnings += 1
                elif severity == 2:
                    self.thread_errors += 1
                elif severity == 3:
                    self.thread_criticals += 1

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
