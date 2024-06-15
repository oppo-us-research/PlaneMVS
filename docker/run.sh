#!/bin/bash
#"""
#* Copyright (c) 2024 OPPO. All rights reserved.
#* Under license: MIT
#* For full license text, see LICENSE file in the repo root
#"""

echo "Current user is : ${USER}"

#hostname=$1
echo "Need your input to define `hostname`"
# define a name to your machine, and this name will be used 
# to name your experiments, so that you can have an idea which
# experiments are run on which machines, if you have many machines 
# to run experiments simutaneously.
read hostname

# Or you can comment the above, and just give a default name as below;
# hostname='rtx3090s1'
echo "Got it hostname=$hostname"

# define some tags to name the Docker image;
USER_NAME=${1:-'ccj'}
GROUP=$(id -gn)
VER=${2:-1.0}
DOCKER_IMAGE=${USER_NAME}/planemvs:$VER

## optional
root_dir=${3:-'/home/ccj/code'}
# user name
u=$(id -un)
# group name
g=$(id -gn)
echo $u $g
echo "DOCKER_IMAGE=$DOCKER_IMAGE"
echo "root_dir=$root_dir"

docker run --runtime=nvidia --gpus all --ipc=host \
    -e HOSTNAME=${hostname} \
    -e ROOT_DIR=${root_dir} \
    -u $USER_NAME:$GROUP \
    -v "/media/disk1:/media/disk1" \
    -v "/home/$u/code:/home/$USER_NAME/code" \
    -v "/home/$u/Downloads:/home/$USER_NAME/Downloads" \
    -v "/media/disk1:/media/disk1" \
    -v "/media/disk2:/media/disk2" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -it $DOCKER_IMAGE /bin/bash
