#!/bin/bash
# Usage ./make.sh -j 4
if [ $# -eq 0 ]
  then
    BUILD_ARGS="-j 2"
else
    BUILD_ARGS="$@"
fi
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)');/workspace/build/third-party/protobuf/lib/cmake" 

cmake --build build --target inference-gpu -- $BUILD_ARGS
cmake --build build --target inference-gpu-throughput -- $BUILD_ARGS
cmake --build build --target install -- $BUILD_ARGS
