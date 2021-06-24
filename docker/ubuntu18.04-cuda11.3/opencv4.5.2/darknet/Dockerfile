# syntax=docker/dockerfile:experimental
FROM joinaero/ubuntu18.04-cuda11.3:opencv4.5.2
LABEL maintainer="join.aero@gmail.com"

ARG cmake_url=https://github.com/Kitware/CMake/releases/download/v3.21.0-rc1/cmake-3.21.0-rc1-linux-x86_64.sh
ARG darknet_url=https://github.com/AlexeyAB/darknet.git

SHELL ["/bin/bash", "-ic"]

RUN apt-get update && apt-get install -y curl \
 \
 && cd /home/ && mkdir cmake/ \
 && curl -O -L ${cmake_url} \
 && sh cmake-*.sh --prefix=/home/cmake --skip-license \
 && rm cmake-*.sh \
 \
 && apt-get remove -y curl && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y build-essential git \
 \
 && cd /home/ \
 && git clone --depth 1 ${darknet_url} \
 \
 && apt-get remove -y git && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

RUN echo "export PATH=/home/cmake/bin\${PATH:+:\${PATH}}" >> ~/.bashrc \
 && echo "export LD_LIBRARY_PATH=/home/opencv-4.5.2/lib:/usr/local/cuda/lib64/stubs\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> ~/.bashrc

RUN source /root/.bashrc \
 && cd /home/darknet/ \
 && mkdir build_release \
 && cd build_release \
 && cmake .. \
 && cmake --build . --target install --parallel `nproc`

WORKDIR /home/darknet
# ENTRYPOINT darknet
CMD ["/bin/bash"]

# docker build -t joinaero/ubuntu18.04-cuda11.3:opencv4.5.2-darknet .
