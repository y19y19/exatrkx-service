FROM nvcr.io/nvidia/tritonserver:22.02-py3
# nvcc version: 11.6 ## nvcc --version
# cudnn version: 8.3.2  ## find / -name "libcudnn*" 2>/dev/null

LABEL description="Ubuntu 20.04 with Acts dependencies for the Exa.TrkX custom backend based on tritonserver"
LABEL maintainer="Haoran Zhao <haoran.zhao@cern.ch>"
# https://github.com/acts-project/machines/blob/master/ubuntu2004_exatrkx/Dockerfile
# https://github.com/xju2/dockers/blob/main/ML/acts-triton/Dockerfile
# ENV DEBIAN_FRONTEND=noninteractive

# Update the CUDA Linux GPG Repository Key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub 

# See also https://root.cern.ch/build-prerequisites
RUN apt-get update -y && apt-get install -y \
    build-essential curl git freeglut3-dev libfreetype6-dev libpcre3-dev\
    libboost-dev libboost-filesystem-dev libboost-program-options-dev libboost-test-dev \
    libtbb-dev ninja-build \
    python3 python3-dev python3-pip python3-numpy \
    rsync zlib1g-dev ccache vim unzip libblas-dev liblapack-dev swig \
    rapidjson-dev \
    libexpat-dev libeigen3-dev libftgl-dev libgl2ps-dev libglew-dev libgsl-dev \
    liblz4-dev liblzma-dev libx11-dev libxext-dev libxft-dev libxpm-dev libxerces-c-dev \
    libzstd-dev ccache \
  && apt-get clean -y

# Manual builds for specific packages
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN  pip3 install --upgrade pip
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib"
ENV PYTHON_PATH="/usr/local/lib"

ENV GET="curl --location --silent --create-dirs"
ENV UNPACK_TO_SRC="tar -xz --strip-components=1 --directory src"
ENV PREFIX="/usr/local"
ENV CUDA_ARCH="80"

# # Append to the bashrc to set the PYTHONPATH
# ENV PYTHONPATH "${PYTHONPATH}:/usr/lib/python3.8/site-packages/"

# Install CMake v3.26.2
RUN cd /tmp && mkdir -p src \
  && ${GET} https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-Linux-x86_64.tar.gz \
    | ${UNPACK_TO_SRC} \
  && rsync -ru src/ ${PREFIX} \
  && cd /tmp && rm -rf /tmp/src  

# Install xxHash v0.7.3
RUN mkdir -p src \
  && ${GET} https://github.com/Cyan4973/xxHash/archive/v0.7.3.tar.gz \
    | ${UNPACK_TO_SRC} \
  && cmake -B build -S src/cmake_unofficial -GNinja\
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
  && cmake --build build -- install -j 20\
  && rm -rf build src

# Install nlohmann/json v3.10.2
RUN mkdir src \
  && ${GET} https://github.com/nlohmann/json/archive/refs/tags/v3.10.2.tar.gz \
    | ${UNPACK_TO_SRC} \
  && cmake -B build -S src -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DJSON_BuildTests=OFF \
  && cmake --build build -- install -j 20\
  && rm -rf build src

# libtorch (unzip cannot be used in a pipe...)
ENV LIBTORCH_URL_GPU="https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu116.zip"
# https://download.pytorch.org/whl/torch_stable.html

RUN ${GET} --output libtorch.zip ${LIBTORCH_URL_GPU} \
  && unzip libtorch.zip \
  && rsync -ruv libtorch/ ${PREFIX} \
  && rm -rf libtorch*

# torchscatter
RUN cd /tmp/ && mkdir src \
  && ${GET} https://github.com/rusty1s/pytorch_scatter/archive/refs/tags/2.0.9.tar.gz \
    | ${UNPACK_TO_SRC} \
  && cmake -B build -S src -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_CUDA_FLAGS=-D__CUDA_NO_HALF_CONVERSIONS__=OFF \
    -DWITH_CUDA=ON \
  && cmake --build build -- install -j 8\
  && rm -rf build src

# root v6.24.06
RUN mkdir src \
  && ${GET} https://root.cern/download/root_v6.24.06.source.tar.gz \
    | ${UNPACK_TO_SRC} \
  && cmake -B build -S src -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -Dfail-on-missing=ON \
    -Dgminimal=ON \
    -Dgdml=ON \
    -Dopengl=ON \
    -Dpyroot=ON \
  && cmake --build build -- install -j 20\
  && rm -rf build src

# cugraph v22.02.00
RUN mkdir src \
  && ${GET} https://github.com/rapidsai/cugraph/archive/refs/tags/v22.02.00.tar.gz \
    | ${UNPACK_TO_SRC} \
  && cmake -B build -S src/cpp -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DBUILD_TESTS=OFF \
    -DBUILD_CUGRAPH_MG_TESTS=OFF \
  && cmake --build build -- install -j 20 \
  && rm -rf build src

# Onnx (download of tar.gz does not work out of the box, since the build.sh script requires a git repository)
RUN git clone https://github.com/microsoft/onnxruntime src \
  && (cd src && git checkout v1.13.1) \
  && ./src/build.sh \
    --config MinSizeRel \
    --build_shared_lib \
    --build_dir build \
    --use_cuda \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/local/cuda \
    --parallel 8 \
    --skip_tests \
    --cmake_extra_defines \
      CMAKE_INSTALL_PREFIX=${PREFIX} \
  && cmake --build build/MinSizeRel -- install -j 20 \
  && rm -rf build src

# faiss v1.7.4
  RUN cd /tmp && mkdir -p src \
  && ${GET} https://github.com/facebookresearch/faiss/archive/refs/tags/v1.7.4.tar.gz \
    | ${UNPACK_TO_SRC} \
  && cd src && mkdir build && cd build \
  && cmake .. -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON \
        -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON \
        -DPython_EXECUTABLE=/usr/bin/python -DPython_LIBRARIES=/usr/lib/python3.8 \
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
  && make -j8 faiss && make -j8 swigfaiss \
  && cd faiss/python && pip3 install . \
  && cd ../.. && make install -j8 && cd .. \
  && rm -rf src

# To build exatrkx_triton, a cpp version of client
# Install grpc
RUN git clone --recurse-submodules -b v1.49.1 --depth 1 https://github.com/grpc/grpc src\
    && cmake -B build -S src -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -- install -j20 \
    && rm -rf src build

# Install triton
RUN git clone https://github.com/triton-inference-server/client.git \
    && cd client && mkdir build && cd build \
    && cmake ../src/c++ -DTRITON_ENABLE_CC_HTTP=OFF \
        -DTRITON_ENABLE_CC_GRPC=ON \
        -DTRITON_ENABLE_PYTHON_GRPC=ON \
        -DCMAKE_PREFIX_PATH="${PREFIX}/lib64/cmake;${PREFIX}/lib/cmake" \
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
        -DTRITON_USE_THIRD_PARTY=OFF  \
   && make -j20 && make install \
   && cd ../.. && rm -rf client