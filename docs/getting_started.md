# Getting Started

## Build using Docker

### [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda)

```bash
docker pull nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
```

```bash
$ docker run --gpus all nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 nvidia-smi
Sun Aug  8 00:00:00 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.100      Driver Version: 440.100      CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 2060    Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   50C    P8     5W /  N/A |    538MiB /  5934MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

Why using CUDA 10.2 here:

1. nvidia-smi, CUDA Version: 10.2
2. PyTorch 1.6.0, CUDA 10.2

<!--
docker run -it --gpus all nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
-->

### [OpenCV](https://github.com/opencv/opencv)

```bash
cd docker/ubuntu18.04-cuda10.2/opencv4.4.0/

docker build \
-t joinaero/ubuntu18.04-cuda10.2:opencv4.4.0 \
--build-arg opencv_ver=4.4.0 \
--build-arg opencv_url=https://gitee.com/cubone/opencv.git \
--build-arg opencv_contrib_url=https://gitee.com/cubone/opencv_contrib.git \
.
```

<!--
#9 99.47 W: GPG error: https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1804/x86_64  Release: The following signatures were invalid: BADSIG F60F4B3D7FA2AF80 cudatools <cudatools@nvidia.com>
#9 99.47 E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release' is not signed.
-->
<!--
#9 1398. [ 63%] Building CXX object modules/xfeatures2d/CMakeFiles/opencv_xfeatures2d.dir/src/boostdesc.cpp.o
#9 1398. /codes/opencv_contrib/modules/xfeatures2d/src/boostdesc.cpp:654:20: fatal error: boostdesc_bgm.i: No such file or directory
#9 1398.            #include "boostdesc_bgm.i"
-->
