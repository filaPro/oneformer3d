FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6

# Install MMCV, MMDetection, MMSegmentation, MMDetection3D
RUN pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html
RUN pip install mmdet==3.0.0
RUN pip install mmsegmentation==1.0.0
RUN pip install git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61

# Install MinkowskiEngine
# Feel free to skip nvidia-cuda-dev if minkowski installation is fine
RUN apt-get -y install libopenblas-dev nvidia-cuda-dev
RUN TORCH_CUDA_ARCH_LIST="6.1 7.0 8.6" pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
    --install-option="--blas=openblas" \
    --install-option="--force_cuda"

# Install python packages
RUN pip install spconv-cu116==2.3.6
RUN pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

# Install ScanNet superpoint segmentator
RUN git clone https://github.com/Karbo123/segmentator.git \
    && cd segmentator/csrc \
    && mkdir build \
    && cd build \
    && cmake .. \
        -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
        -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
        -DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` \
    && make \
    && make install \
    && cd ../../..
