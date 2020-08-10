# syntax=docker/dockerfile:experimental
FROM joinaero/ubuntu18.04-cuda10.2:opencv4.4.0
LABEL maintainer="join.aero@gmail.com" \
      joinaero.release-date="2020-08-10" \
      joinaero.version="0.1.0"

ARG cmake_url=https://github.com/Kitware/CMake/releases/download/v3.18.1/cmake-3.18.1-Linux-x86_64.sh
ARG darknet_url=https://github.com/AlexeyAB/darknet.git

SHELL ["/bin/bash", "-ic"]

RUN mv /etc/apt/sources.list.d/*.list /home/ \
 && apt update && apt install -y curl \
 \
 && cd /home/ && mkdir cmake/ \
 && curl -O -L ${cmake_url} \
 && sh cmake-*.sh --prefix=/home/cmake --skip-license \
 && rm cmake-*.sh \
 \
 && mv /home/*.list /etc/apt/sources.list.d/ \
 && apt remove -y curl && apt autoremove -y \
 && rm -rf /var/lib/apt/lists/*

RUN echo "export PATH=/usr/local/cuda/bin:/home/cmake/bin:\$PATH" >> ~/.bashrc \
 && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/opencv-4.4.0/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

RUN mv /etc/apt/sources.list.d/*.list /home/ \
 && apt update && apt install -y build-essential git \
 \
 && cd /home/ \
 && git clone --depth 1 ${darknet_url} \
 && cd /home/darknet/ \
 && ./build.sh \
 \
 && mv /home/*.list /etc/apt/sources.list.d/ \
 && apt remove -y git && apt autoremove -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /home/darknet
# ENTRYPOINT darknet
CMD ["/bin/bash"]
