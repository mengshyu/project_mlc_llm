#!/bin/bash
set -e

ROOT=$PWD
REPO_URL="https://github.com/mlc-ai/mlc-llm"
REPO_NAME=$(basename "$REPO_URL" | sed 's/\.git$//')
MLC_LLM_ROOT=$ROOT/$REPO_NAME
BUILD_DIR=$MLC_LLM_ROOT/build
PYTHON_DIR=$MLC_LLM_ROOT/python

VENV_ACT=~/workspace/venv/bin/activate

function Build()
{

    if [ ! -d $MLC_LLM_ROOT ];then
        echo "$MLC_LLM_ROOT doest not exist!"
        exit 1
    fi

    if [ -d $BUILD_DIR ];then
        rm -rf $BUILD_DIR
    fi

    mkdir -p $BUILD_DIR

    cd $BUILD_DIR

    python3 ../cmake/gen_cmake_config.py
    echo 'set(USE_THRUST OFF)' >> config.cmake
    cmake .. && cmake --build . --parallel $(nproc) && cd ..
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


echo "Project name: $REPO_NAME"

if [ ! -d $MLC_LLM_ROOT ];then
    git clone $REPO_URL --recursive
fi

Build
Install

