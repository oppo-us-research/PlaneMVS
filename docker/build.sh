#!/bin/bash
#"""
#* Copyright (c) 2024 OPPO. All rights reserved.
#* Under license: MIT
#* For full license text, see LICENSE file in the repo root
#"""

#> See: Verify the version of ubuntu running in a Docker container,
#> at https://stackoverflow.com/questions/38003194/verify-the-version-of-ubuntu-running-in-a-docker-container;

#It's simple variables that are shell script friendly, so you can run

echo "------------------->"
echo "------------------->"
echo "----start building docker image ..."

PYTHON_VERSION=3.10

USER_NAME=${1:-'ccj'}
VER=${2:-1.0}
HOST_USER_NAME=${3:-'changjiang'}
DOCKER_TAG=${USER_NAME}/planemvs:$VER

echo "Will build docker container $DOCKER_TAG ..."
#exit

#[ ! -d "./files" ] && mkdir ./files
#/bin/cp ./bashrc_extra ./files/
#/bin/cp ./dev_requirements.txt ./files/
#exit

docker build --tag $DOCKER_TAG \
    --force-rm \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    --build-arg USER_NAME=$USER_NAME \
    --build-arg GROUP_NAME=$(id -gn) \
    --build-arg python=${PYTHON_VERSION} \
    --build-arg HOST_USER_NAME=${HOST_USER_NAME} \
    -f Dockerfile .

#rm -r ./files
# could add:
# --no-cache \
#--force-rm \
