#!/bin/bash
cmake -S . -B build \
 -DCMAKE_C_COMPILER=`which gcc` \
 -DCMAKE_CXX_COMPILER=`which g++` \
 -DCMAKE_INSTALL_PREFIX:PATH=install \


 cmake --build build --target install