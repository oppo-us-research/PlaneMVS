#"""
#* Copyright (c) 2024 OPPO. All rights reserved.
#* Under license: MIT
#* For full license text, see LICENSE file in the repo root
#"""

# to work for pre-compiled pytorch (torch==2.2.2 with cuda 12.1, see https://download.pytorch.org/whl/cu121)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

## > see this Dockerfile example at https://github.com/TRI-ML/dgp/blob/master/Dockerfile;
ARG python=3.10

ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

# Temporary: due to GPG error "public key is not available" in Ubuntu 20.04 CUDA 11.4.0
# > see: https://github.com/NVIDIA/nvidia-docker/issues/1632#issuecomment-1112667716
#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Using -y is convenient to automatically answer yes to all the questions;
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    git \
    build-essential \
    graphviz \
    cmake curl \
    libpng-dev \
    libjpeg-dev libjpeg-dev \
    libgl1-mesa-dev \
    ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    vim tmux \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    # required by OpenCV3
    libgtk2.0-0 libcanberra-gtk-module \
    python${PYTHON_VERSION}-tk \
    # eth3d dataset etc point cloud evaluation
    libboost-all-dev libeigen3-dev libpcl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    /usr/bin/python3 get-pip.py && \
    rm get-pip.py

#COPY files/* /tmp/
COPY dev_requirements.txt /tmp/
#COPY bashrc_extra /tmp/

RUN pip install --upgrade pip
#RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# Or use this PyTorch version, which has also be verified by our code;

RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install -r /tmp/dev_requirements.txt 
## Or, do not install here, instead to install them once you 
#log into the Docker container;

RUN pip install ipython

# Create a non-root user and switch to it
ARG USER_ID=1000
ARG GROUP_ID=1000

ARG USER_NAME='dummy'
ARG GROUP_NAME='dummy'

##--- Or, change those to first_run.sh
RUN addgroup --gid $GROUP_ID $GROUP_NAME
# Create a user 'appuser' under 'xyzgroup'
RUN useradd -rm -d /home/$USER_NAME --shell /bin/bash  --uid $USER_ID --gid $GROUP_ID -G $GROUP_NAME $USER_NAME

EXPOSE 22
USER $USER_NAME:$GROUP_NAME
WORKDIR /home/$USER_NAME

RUN pip install pycocotools
# install pycocotools
# > see: https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/docker/Dockerfile
#RUN git clone https://github.com/cocodataset/cocoapi.git \
# && cd cocoapi/PythonAPI \
# && /usr/bin/python3 setup.py build_ext install

RUN pip install cityscapesscripts
# install cityscapesScripts
#RUN git clone https://github.com/mcordts/cityscapesScripts.git \
# && cd cityscapesScripts \
# && /usr/bin/python3 setup.py build_ext install

##################################################
###  Personalized System initialization commands/installations (Optional)
##################################################
### (Optional) Create softlinks ###
ARG HOST_USER_NAME='changjiang'
RUN ln -s /home/${HOST_USER_NAME}/code /home/${USER_NAME}/code
#RUN ln -s /home/${HOST_USER_NAME}/Downloads /home/${USER_NAME}/Downloads

##################################################
###  Environment variable setup
##################################################
ENV PATH=/home/$USER_NAME/.local/bin:$PATH
ENV PATH=/home/$USER_NAME/bin:$PATH

### color prompt ###
ENV TERM='xterm-color'

##################################################
###  Bashrc
##################################################
### (Optional) bashrc ###
COPY bashrc_extra /home/$USER_NAME/.bashrc_extra
RUN echo "source ~/.bashrc_extra" >> ~/.bashrc
