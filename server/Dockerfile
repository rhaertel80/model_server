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

# This is the online-prediction Docker image on which the user's Google Cloud ML
# model is served, and should be placed at
# gcr.io/cloud-ml-service-gcr-public/cloud-ml-online-prediction-cpu
#
# To run this image locally, e.g.:
# export IMAGE_LABEL=cloud_ml_v2_20161210_1206_RC00
# export IMAGE_PATH="gcr.io/cloud-ml-service-public/cloud-ml-online-prediction-cpu:${IMAGE_LABEL}"
# export MODEL_PATH="$(pwd)/third_party/py/google/cloud/ml/docker/training_and_prediction/online_prediction_cpu/testdata/string_model"
# gcloud docker -- pull ${IMAGE_PATH}
# gcloud docker -- run --rm -p 8080:8080 --name "local_prediction_server" -t \
#     -v ${MODEL_PATH}:/tmp/mounted_model \
#     -e model_path=/tmp/mounted_model \
#     -e chemist_endpoint=servicecontrol.googleapis.com \
#     -e cloud_service_name=ml.googleapis.com \
#     -e consumer_project_number=1 \
#     -e version_reporting_id=fake-version-id \
#     -e prediction_engine=TF_SESSION_RUN \
#      ${IMAGE_PATH}

FROM cloud-ml-tensorflow-serving-cpu

COPY version-config.json /tmp/version-config.json

# Install Cloud ML SDK.
# We copy the latest over, in case it's referred to by sdk-version.txt,
# but we only actually install it for the HEAD container.

# We are copying multiple sdk tars but eventually install only one based on the
# online_prediction_sdk_version defined in version-config.json.
COPY cloudml.tar.gz /tmp/cloudml.tar.gz
COPY cloudml.nodeps.tar.gz /tmp/cloudml.nodeps.tar.gz

# Create packages directory if not created as part of tensorflow_serving_cpu
RUN mkdir -p /opt/google/cloud/ml/packages

# TODO(b/34465797): Pin the SDK version for online prediction.
# Online prediction currently relies on recent changes to
# google/cloud/ml/prediction and won't work with version 0.1.7 of the SDK, so
# we currently always install the HEAD version in /tmp/cloudml.tar.gz instead
# of the version specified in version-config.json.
RUN curl -o /opt/google/cloud/ml/packages/cloudml.tar.gz $(cat /tmp/version-config.json | python -c "import sys,yaml; print yaml.load(sys.stdin)['online_prediction_sdk_version']") && \
    pip install /opt/google/cloud/ml/packages/cloudml.tar.gz && \
    rm /opt/google/cloud/ml/packages/cloudml.tar.gz && \
    rm /tmp/cloudml.tar.gz && \
    rm /tmp/cloudml.nodeps.tar.gz && \
    # Remove pip cache.
    rm -rf /root/.cache/pip

# Install Online Prediction's required pip packages.
RUN pip install WebOb==1.7.2 Paste==2.0.3 tornado==4.5.1 grpcio==1.3.0 \
    requests==2.13.0 webapp2==3.0.0b1 gcloud==0.18.3 SciPy==0.19.1 \
    scikit-learn==0.19.0

# Reinstall numpy from source as a workaround for b/34886296
# NB: this is already being done in the dependencies layers, but we need
# to insure it's not being undone by any of the above (b/70046917).
RUN yes | pip uninstall numpy && \
    pip install numpy==1.12.1 --no-binary=:all: -I

# Install the xgboost version specified in version-config.json
# TODO(b/69132328): Installing from github might slow things down. Consider having a
# pre-packaged xgboost release from github and copy that instead of building
# everytime.
RUN xgboost_version=$(cat /tmp/version-config.json | python -c "import sys,yaml; print yaml.load(sys.stdin)['xgboost_version']") && \
  echo "xgboost_version: ${xgboost_version}" && \
  if [ "${xgboost_version}" = "head" ]; then \
    xgboost_commit=$(cat /tmp/version-config.json | python -c "import sys,yaml; print yaml.load(sys.stdin)['xgboost_commit']") && \
    echo "Installing xgboost from github with commit id: ${xgboost_commit}" && \
    mkdir -p "/tmp/xgboost" && \
    cd "/tmp/xgboost" && \
    git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    git reset --hard ${xgboost_commit} && \
    make -j4 && \
    cd python-package && \
    python setup.py install && \
    cd ../../ && \
    export PYTHONPATH=$PYTHONPATH:./xgboost/python-package; \
  else \
      echo "Installing xgboost version ${xgboost_version}" && \
      pip install xgboost==${xgboost_version}; \
  fi


# See https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#protobuf-library-related-issues
RUN pip install --upgrade protobuf==3.2.0 && \
    # Check that the fast implementation of protobuf is used.
    python -c "from google.protobuf.internal import api_implementation; assert api_implementation._default_implementation_type == 'cpp'; print 'Verified fast protobuf used.'" && \
    # Remove pip cache.
    rm -rf /root/.cache/pip

# Unpack prediction server.
COPY opensource_prediction_server.tar /tmp/opensource_prediction_server.tar
RUN tar -C / -xvf /tmp/opensource_prediction_server.tar && \
    rm /tmp/opensource_prediction_server.tar

# Setup for running gsutil. Point both config directories to tmp because all
# other directories are not writeable by the client app.
ENV CLOUDSDK_CONFIG /tmp
# GSUtil state_dir parameter points to temporary scratch space gsutil can use.
# It has to be in /tmp to be writeable in gvisor/titanium. Boto is documented here
# https://cloud.google.com/storage/docs/boto-gsutil
RUN echo '[GSUtil]\nstate_dir = /tmp' >> /etc/boto.cfg

# Only expose the prediction server port.
EXPOSE 8080

# TODO(b/36191865): We cannot change the ENTRYPOINT for this docker container
# because the current rollout policy doesn't allow it. Therefore, while building
# the container for 1.0 runtimes, we will copy the code from prediction_server.py
# into the prediction_server_beta.py. We will leave the _lib as such.
RUN export CLOUDML_RUNTIME_VERSION=$(python -c 'import yaml;print yaml.load(open("/tmp/version-config.json", "r"))["cloudml_runtime_version"]') && \
    echo "Cloud ML Runtime version ${CLOUDML_RUNTIME_VERSION}" && \
    cat prediction_server.py > prediction_server_beta.py && \
    rm prediction_server.py
# Allow the prediction server to access the TensorFlow, Scikit-learn, and
# Xgboost libraries in the frameworks/ directory.
RUN touch __init__.py
RUN touch frameworks/__init__.py

RUN rm /tmp/version-config.json
# Startup.
# Note: Until b/36191865 is resolved, we cannot change the entrypoint for this docker
# container.
ENTRYPOINT ["python", "prediction_server_beta.py", "--model_server_binary_path", "/bin/tensorflow_model_server"]
