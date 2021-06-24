# syntax=docker/dockerfile:experimental
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
LABEL maintainer="join.aero@gmail.com"

ARG opencv_ver=4.5.2
ARG opencv_url=https://github.com/opencv/opencv.git
ARG opencv_contrib_url=https://github.com/opencv/opencv_contrib.git
ARG pip_index_url=https://mirrors.aliyun.com/pypi/simple/

# COPY sources.list /etc/apt/

ENV CUDA_HOME /usr/local/cuda

RUN apt-get update \
 && apt-get install -y build-essential cmake git libgtk-3-dev python3-pip \
 \
 && if [ -n "${pip_index_url}" ]; then \
      pip3 install numpy -i ${pip_index_url}; \
    else \
      pip3 install numpy; \
    fi \
 \
 && mkdir -p /codes && cd /codes/ \
 \
 && git clone -b ${opencv_ver} --depth 1 ${opencv_url} \
 && git clone -b ${opencv_ver} --depth 1 ${opencv_contrib_url} \
 \
 && cd /codes/opencv/ \
 && mkdir _build && cd _build/ \
 && cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/home/opencv-${opencv_ver} \
    -DOPENCV_EXTRA_MODULES_PATH=/codes/opencv_contrib/modules \
    \
    -DWITH_CUDA=ON \
    \
    -DBUILD_DOCS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    .. \
 && make -j$(nproc) \
 && make install \
 \
 && rm -rf /codes/ \
 \
 && apt-get remove -y cmake git && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

ENV OpenCV_DIR /home/opencv-${opencv_ver}/lib/cmake

CMD ["/bin/bash"]

# docker build -t joinaero/ubuntu18.04-cuda11.3:opencv4.5.2 .
