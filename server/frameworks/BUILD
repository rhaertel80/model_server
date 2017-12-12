# Description:
#   Docker image used by Cloud ML online prediction (on CPUs).

package(default_visibility = [
    "//third_party/py/google/cloud/ml/docker:__subpackages__",
])

licenses(["notice"])  # Apache License 2.0

py_library(
    name = "tf_prediction_server_lib",
    srcs = [
        ":tf_prediction_server_lib.py",
    ],
    deps = [
    ],
)

py_test(
    name = "tf_prediction_server_lib_test",
    srcs = [
        "tf_prediction_server_lib.py",
        "tf_prediction_server_lib_test.py",
    ],
    data = [
        "//third_party/py/google/cloud/ml/testdata:half_plus_two_discrete_multiple_inputs_multiple_outputs_saved_model",
        "//third_party/py/google/cloud/ml/testdata:half_plus_two_examples_multiple_inputs_saved_model",
    ],
    deps = [
        "//pyglib",
        "//pyglib:gfile",
        "//pyglib/flags",
        "//testing/pybase",
        "//third_party/py/gcloud:alternative_gcloud",
        "//third_party/py/google/cloud/ml",
        "//third_party/py/numpy",
        "//third_party/py/requests",
        "//third_party/py/sklearn",
        "//third_party/py/tensorflow",
        "//third_party/py/webapp2",
        "//third_party/py/webtest",
        "//third_party/tensorflow/core:protos_all_py_pb2",
        "//third_party/tensorflow_serving/apis:predict_proto_py_pb2",
        "//third_party/tensorflow_serving/apis:prediction_service_proto_py_pb2",
    ],
)
