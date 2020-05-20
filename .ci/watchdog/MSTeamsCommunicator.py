#!/usr/bin/python3

# INTEL CONFIDENTIAL
# Copyright 2018-2020 Intel Corporation
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
import requests


class MSTeamsCommunicator:
    """Class communicating with MSTeams using Incoming Webhook.
    The purpose of this class is to use MSTeams API to send message.
    Docs for used API, including wrapped methods can be found at:
    https://docs.microsoft.com/en-us/outlook/actionable-messages/send-via-connectors
    """

    def __init__(self, _ci_alerts_channel_url):
        self._ci_alerts_channel_url = _ci_alerts_channel_url
        self._queued_messages = {
            self._ci_alerts_channel_url: []
        }

    @property
    def messages(self):
        """
        Get list of queued messages.

            :return:           List of queued messages
            :return type:      List[String]
        """
        return self._queued_messages.values()

    def queue_message(self, message):
        """
        Queue message to be sent later.

            :param message:     Message content
            :type message:      String
        """
        self._queued_messages[self._ci_alerts_channel_url].append(message)

    def _parse_text(self, message):
        """
        Parse text to display as alert.

            :param message:     Unparsed message content
            :type message:      String
        """
        message_split = message.split("\n")
        title = message_split[2]
        log_url = message_split[-1]
        text = message_split[3]
        header = message_split[0].split(" - ")
        header_formatted = '{} - [Watchdog Log]({})'.format(header[0], header[1])
        text_formatted = "{}: ***{}***".format(text.split(":", 1)[0], text.split(":", 1)[1])

        return title, log_url, '{}\n\n{}'.format(header_formatted, text_formatted)

    def _json_request_content(self, title, log_url, text_formatted):
        """
        Create final json request to send message to MS Teams channel.

            :param title:            Title of alert
            :param log_url:          URL to Watchdog log
            :param text_formatted:   General content of alert - finally formatted
            :type title:             String
            :type title:             String
            :type title:             String
        """
        data = {
            "@context": "https://schema.org/extensions",
            "@type": "MessageCard",
            "themeColor": "0072C6",
            "title": title,
            "text": text_formatted,
            "potentialAction":
                [
                    {
                        "@type": "OpenUri",
                        "name": "Open Log",
                        "targets":
                            [
                                {
                                    "os": "default",
                                    "uri": log_url
                                }
                            ]
                    }
                ]
        }
        return data

    def _send_to_channel(self, message, channel_url):
        """
        Send MSTeams message to specified channel.

            :param message:            Message content
            :type message:             String
            :param channel_url:        Channel url
            :type channel_url:         String
        """
        title, log_url, text_formatted = self._parse_text(message)
        data = self._json_request_content(title, log_url, text_formatted)

        try:
            requests.post(url=channel_url, json=data)
        except Exception as ex:
            raise Exception('!!CRITICAL!! MSTeamsCommunicator: Could not send message '
                            'due to {}'.format(ex))

    def send_message(self, message, quiet=False):
        """
        Send queued messages as single communication.

            :param message:     Final message's content
            :param quiet:       Flag for disabling sending report through MS Teams
            :type message:      String
            :type quiet:        Boolean
        """
        for channel, message_queue in self._queued_messages.items():
            final_message = message + '\n\n' + '\n'.join(message_queue)
            if not quiet and message_queue:
                self._send_to_channel(final_message, channel)
