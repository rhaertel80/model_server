"""Tests for google3.cloud.ml.beta.prediction.metric_reporting_lib."""

import datetime
import json
import time

import google3

import mock
import requests

from google.cloud.ml.docker.training_and_prediction.online_prediction import metric_reporting_lib

from google3.testing.pybase import googletest

PROJECT_NUMBER = 827153953634
REPORTING_INTERVAL_SEC = 2


@mock.patch("requests.post")
class MetricReportingLibTest(googletest.TestCase):

  def tearDown(self):
    metric_reporting_lib.stop_reporting()

  def do_test_reporting(self, expected_bearer, mock_post):
    test_start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    mock_post.return_value.status_code = requests.codes.ok
    mock_post.return_value.json.return_value = {}
    metric_reporting_lib.start_reporting("servicecontrol.googleapis.com",
                                         "ml.googleapis.com", PROJECT_NUMBER,
                                         "version_123", REPORTING_INTERVAL_SEC)
    time.sleep(REPORTING_INTERVAL_SEC * 3 + 0.3)
    self.assertEqual(3, mock_post.call_count)

    # Examine the last call to Report for the uptime.
    (url, body_text), kwargs = mock_post.call_args
    self.assertEqual("https://servicecontrol.googleapis.com/v1/"
                     "services/ml.googleapis.com:report", url)
    self.assertEqual({
        "Authorization": "Bearer %s" % expected_bearer
    }, kwargs["headers"])
    body = json.loads(body_text)
    self.assertEqual(1, len(body["operations"]))
    operation = body["operations"][0]
    self.assertEqual("online_prediction", operation["operationName"])
    self.assertEqual("project_number:827153953634", operation["consumerId"])
    self.assertEqual(1, len(operation["metricValueSets"]))
    self.assertLess(test_start_time, operation["startTime"])
    self.assertLess(operation["startTime"], operation["endTime"])
    metric_value_set = operation["metricValueSets"][0]
    self.assertEqual("ml.googleapis.com/online_prediction_node_second",
                     metric_value_set["metricName"])
    self.assertEqual(1, len(metric_value_set["metricValues"]))
    metric_value = metric_value_set["metricValues"][0]
    self.assertEqual(str(REPORTING_INTERVAL_SEC), metric_value["int64Value"])
    self.assertLess(test_start_time, metric_value["startTime"])
    self.assertLess(metric_value["startTime"], metric_value["endTime"])
    self.assertEqual(
        str(REPORTING_INTERVAL_SEC), metric_value["labels"]["/seconds"])
    self.assertEqual(datetime.datetime.now().strftime("%Y/%m/%d"),
                     metric_value["labels"]["/consumption_date"])
    self.assertEqual("1", metric_value["labels"]["/workers"])
    self.assertEqual("version_123",
                     metric_value["labels"]["/version_reporting_id"])

  @mock.patch("requests.get")
  def testReportingWithMetadataServiceBearer(self, mock_get, mock_post):
    mock_get.return_value.status_code = requests.codes.ok
    mock_get.return_value.json.return_value = {
        "access_token": "METADATA_SERVICE_BEARER"
    }

    self.do_test_reporting("METADATA_SERVICE_BEARER", mock_post)

    (url,), kwargs = mock_get.call_args
    self.assertEqual(
        "http://metadata.google.internal/computeMetadata/v1/instance/"
        "service-accounts/default/token", url)
    self.assertEqual({"Metadata-Flavor": "Google"}, kwargs["headers"])

  def testReportingWithSetAuthorizationBearer(self, mock_post):
    metric_reporting_lib.set_authorization_bearer("SET_BEARER")
    self.do_test_reporting("SET_BEARER", mock_post)

if __name__ == "__main__":
  googletest.main()
