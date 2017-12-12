# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A module for reporting metrics to Google Service Control.

Once initialized, the module will continuously report the model up-time
metric to Google Service Control.

The caller can supply auth bearer tokens to the module by calling
set_authorization_bearer().

If no auth bearer token is set, the module will use the auth bearer token from
the metadata service.

If the auth bearer token cannot be retrieved, the reporting fails.
"""

# TODO(b/33358770): Remove the files here once we are ready to migrate to the
# new docker container for billing.

import datetime
import json
import logging
import threading
import time
import uuid

import google3

import requests

URL_TEMPLATE = "https://%(endpoint)s/v1/services/%(service_name)s:report"
METRIC_REPORTING_INTERVAL_SEC = 30
OPERATION_NAME = "online_prediction"
INSTANCE_METRIC_NAME = "ml.googleapis.com/online_prediction_instance"
NODE_SECOND_METRIC_NAME = "ml.googleapis.com/online_prediction_node_second"
CONSUMPTION_DATE_LABEL = "/consumption_date"
WORKERS_LABEL = "/workers"
SECONDS_LABEL = "/seconds"
VERSION_REPORTING_ID_LABEL = "/version_reporting_id"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
TOKEN_REQUEST_URL = ("http://metadata.google.internal/computeMetadata/v1"
                     "/instance/service-accounts/default/token")
TOKEN_REQUEST_HEADERS = {"Metadata-Flavor": "Google"}
HTTP_TIMEOUT_SEC = 10.0


def get_access_token_from_metadata_service():
  """Retrieves an auth bearer token from the metadata service."""
  http_response = requests.get(TOKEN_REQUEST_URL, headers=TOKEN_REQUEST_HEADERS,
                               timeout=HTTP_TIMEOUT_SEC)
  if http_response.status_code != requests.codes.ok:
    logging.error("The metadata service returned HTTP code %d: %s",
                  http_response.status_code, http_response.text)
    return None
  token_response = http_response.json()
  if "access_token" not in token_response:
    logging.error("Unexpected response from the metadata service: %s",
                  json.dumps(token_response))
    return None
  return token_response["access_token"]


class _ServiceControlClient(object):
  """A simple REST client for the Service Control API."""

  def __init__(self, endpoint, ml_service_name, project_number):
    """Constructor.

    Args:
      endpoint: the Service Control endpoint,
                e.g "servicecontrol.googleapis.com"
      ml_service_name: Cloud ML service name, e.g "ml.googleapis.com"
      project_number: the customer project number, e.g 123456789
    """
    self._endpoint = endpoint
    self._ml_service_name = ml_service_name
    self._project_number = project_number

  def report_metric(self, start_time, end_time, metric_name, value, labels):
    """Reports a metric to Service Control.

    Args:
      start_time: datetime.datetime object representing the period start time.
      end_time: datetime.datetime object representing the period end time.
      metric_name: the name of the metric registered in Service Control.
      value: integer metric value to be reported.
      labels: a string-to-string dictionary with labels as
              configured in Service Control for this metric.
    """

    url = URL_TEMPLATE % {"endpoint": self._endpoint,
                          "service_name": self._ml_service_name}
    operation = {
        "operationId": str(uuid.uuid4()),
        "operationName": OPERATION_NAME,
        "consumerId": "project_number:%d" % self._project_number,
        "startTime": start_time.strftime(TIMESTAMP_FORMAT),
        "endTime": end_time.strftime(TIMESTAMP_FORMAT),
        "metricValueSets": [{
            "metricName": metric_name,
            "metricValues": [{
                # Not a mistake: value needs to be string.
                "int64Value": str(value),
                "startTime": start_time.strftime(TIMESTAMP_FORMAT),
                "endTime": end_time.strftime(TIMESTAMP_FORMAT),
                "labels": labels,
            }],
        }],
    }
    report_request = {"operations": [operation]}

    # TODO(b/33276769): remove when migration to the new design is done.
    if _current_auth_bearer:
      auth_bearer = _current_auth_bearer
    else:
      auth_bearer = get_access_token_from_metadata_service()
      if not auth_bearer:
        logging.error("Could not retrieve an access token.")
        return
    headers = {"Authorization": "Bearer " + auth_bearer}

    http_response = requests.post(
        url, json.dumps(report_request), headers=headers,
        timeout=HTTP_TIMEOUT_SEC)
    if http_response.status_code != requests.codes.ok:
      logging.error("ServiceControl Report() returned HTTP Code %d: %s",
                    http_response.status_code, http_response.text)
      return
    report_response = http_response.json()
    if report_response.get("reportErrors", []):
      logging.error("ServiceControl Report() returned errors: %s",
                    json.dumps(report_response["reportErrors"]))
      return


# TODO(b/33276769): delete when migration to the new design is done.
_current_auth_bearer = None

# If set to true, makes the reporting thread exit.
_stop_requested = False
# A thread object representing the reporting thread.
_reporting_thread = None


# TODO(b/33276769): delete when migration to the new design is done.
def set_authorization_bearer(bearer):
  """Sets the latest authorization access token.

  The access token must be valid to report metrics for the Cloud ML producer
  project to Service Control.

  Args:
    bearer: string with the OAuth2 auth bearer token.
  """
  global _current_auth_bearer
  _current_auth_bearer = bearer


def start_reporting(endpoint,
                    ml_service_name,
                    project_number,
                    version_reporting_id,
                    reporting_interval_sec=METRIC_REPORTING_INTERVAL_SEC):
  """Starts metric reporting in a separate thread and returns immediately.

  If a reporting thread was already running, it's stopped first and a new
  thread is started instead with the given parameters.

  Args:
      endpoint: the Service Control endpoint,
                e.g "servicecontrol.googleapis.com"
      ml_service_name: Cloud ML service name, e.g "ml.googleapis.com"
      project_number: the customer project number, e.g 123456789
      version_reporting_id: the reporting id of the current version.
      reporting_interval_sec: the interval for reporting, in seconds.
  Raises:
      ValueError: if one of the arguments is empty.
  """
  if not endpoint:
    raise ValueError("ServiceControl endpoint must be provided.")
  if not ml_service_name:
    raise ValueError("Cloud ML service name must be provided.")
  if not project_number:
    raise ValueError("The user project number must be provided.")
  if not version_reporting_id:
    raise ValueError("The version reporting id must be provided.")
  logging.info("Starting the metric reporting thread: "
               "ServiceControl endpoint %s, ML endpoint %s, "
               "project number %d, version_reporting_id %s", endpoint,
               ml_service_name, project_number, version_reporting_id)
  global _stop_requested
  global _reporting_thread
  if _reporting_thread:
    stop_reporting()
  _reporting_thread = threading.Thread(
      target=_reporting_worker,
      args=(endpoint, ml_service_name, project_number, version_reporting_id,
            reporting_interval_sec))
  _stop_requested = False
  _reporting_thread.start()


def stop_reporting():
  """Stops the reporting thread.

  If a reporting thread was running, tells it to stop and returns when the
  thread stopped. Otherwise does nothing.
  """
  global _stop_requested
  global _reporting_thread
  if _reporting_thread:
    _stop_requested = True
    _reporting_thread.join()
    _reporting_thread = None


def _reporting_worker(endpoint, ml_service_name, project_number,
                      version_reporting_id, reporting_interval_sec):
  """Periodically reports metrics to ServiceControl until stop is requested."""
  service_control_client = _ServiceControlClient(endpoint, ml_service_name,
                                                 project_number)
  previous_time_sec = datetime.datetime.now()
  while not _stop_requested:
    logging.debug("Sleeping for %d seconds.", reporting_interval_sec)
    time.sleep(reporting_interval_sec)
    current_time_sec = datetime.datetime.now()
    seconds_to_report = int((current_time_sec - previous_time_sec
                            ).total_seconds())
    consumption_date = current_time_sec.strftime("%Y/%m/%d")
    logging.debug("Reporting metrics to Chemist.")
    service_control_client.report_metric(
        previous_time_sec, current_time_sec, NODE_SECOND_METRIC_NAME,
        seconds_to_report, {
            CONSUMPTION_DATE_LABEL: consumption_date,
            WORKERS_LABEL: "1",
            SECONDS_LABEL: str(seconds_to_report),
            VERSION_REPORTING_ID_LABEL: version_reporting_id,
        })
    previous_time_sec = current_time_sec
  logging.info("Stop requested, exiting the reporting thread.")
