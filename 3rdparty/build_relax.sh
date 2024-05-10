#!/bin/bash
set -e

ROOT=$PWD
REPO_URL="git@github.com:mengshyu/relax.git"
#REPO_URL="https://github.com/mengshyu/relax.git"
REPO_NAME=$(basename "$REPO_URL" | sed 's/\.git$//')
RELAX_ROOT=$ROOT/$REPO_NAME
BUILD_DIR=$RELAX_ROOT/build
PYTHON_DIR=$RELAX_ROOT/python
LLVM_CONFIG=llvm-config-15
VENV_ACT=~/workspace/venv/bin/activate


function Build()
{

    if [ ! -d $RELAX_ROOT ];then
        echo "$RELAX_ROOT doest not exist!"
        exit 1
    fi

    rm -rf $BUILD_DIR && mkdir $BUILD_DIR && cd $BUILD_DIR

    cd $BUILD_DIR

    cp ../cmake/config.cmake .

    # controls default compilation flags
    echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
    # LLVM is a must dependency
    echo "set(USE_LLVM \"$LLVM_CONFIG --ignore-libllvm\")" >> config.cmake
    echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
    # GPU SDKs, turn on if needed
    echo "set(USE_CUDA   /usr/local/cuda)" >> config.cmake
    echo "set(USE_CUDA  ON)" >> config.cmake
    echo "set(USE_CUBLAS ON)" >> config.cmake
    #echo "set(USE_CUTLASS OFF)" >> config.cmake
    echo "set(USE_METAL  OFF)" >> config.cmake
    echo "set(USE_VULKAN OFF)" >> config.cmake
    echo "set(USE_OPENCL ON)" >> config.cmake
    echo "set(CMAKE_CUDA_ARCHITECTURES 75)" >> config.cmake
    # FlashInfer related, requires CUDA w/ compute capability 80;86;89;90
    #echo "set(USE_FLASHINFER ON)" >> config.cmake
    #echo "set(FLASHINFER_CUDA_ARCHITECTURES 89)" >> config.cmake
    #echo "set(USE_THRUST OFF)" >> config.cmake

    #Qualcomm Hexagon
    #echo "set(USE_HEXAGON ON)" >> config.cmake
    #echo "set(USE_HEXAGON_SDK /local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.0.1)" >> config.cmake
    #echo "set(USE_HEXAGON_RPC ON)" >> config.cmake
    #echo 'set(USE_HEXAGON_ARCH "v73")' >> config.cmake

    cmake  .. && cmake --build . --parallel $(nproc)
    #cmake -DCMAKE_CXX_COMPILER="/home/msyu/workspace/project_tvm_llm/3rdparty/llvm-project/build/bin/clang-17"  .. && cmake --build . --parallel $(nproc)
}

function Install()
{
    if [ ! -f $VENV_ACT ];then
        echo "python venv doest not install!"
        exit 1
    fi
    source $VENV_ACT
    cd $PYTHON_DIR
    pip install -e .
}


if [ ! -d $RELAX_ROOT ];then
    git clone $REPO_URL --recursive
fi

Build
Install

