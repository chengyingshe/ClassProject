#!/bin/bash

source /home/scy/anaconda3/etc/profile.d/conda.sh

venv_name=oneformer3d

if ! conda info -e | grep -q "^$venv_name\s"; then
    conda create -n $venv_name python=3.8 -y
fi

conda activate $venv_name

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.txt

sudo apt install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6

pip install --no-deps mmengine==0.7.3 \
    mmdet==3.0.0 \
    mmsegmentation==1.0.0 \
    git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61

pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html --no-deps

sudo apt install -y build-essential python3-dev libopenblas-dev

pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
 
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-1.13.0+cu116.html --no-deps

# install segmentator
cd ..

if [ ! -d "segmentator" ]; then
    git clone https://github.com/Karbo123/segmentator.git
    cd segmentator/csrc && mkdir -p build && cd build
    cmake .. \
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
    -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    -DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` 

    make -j4 && make install
    cd ../../..
fi

# Open3D (visualization, optional)
pip install open3d

# install flash attention
pip install flash-attn --no-build-isolation