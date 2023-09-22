FROM hrzhao076/custom_backend:2.0
LABEL description="the Exa.TrkX custom backend based on tritonserver, including backend library"
LABEL maintainer="Haoran Zhao <haoran.zhao@cern.ch>"
LABEL version="3.0"

# Install dependencies
RUN apt update && apt install -y time tree && apt clean -y 
RUN pip install -U pandas matplotlib seaborn 

# Copy the source code
COPY . /src/

# Build exatrkx_cpu 
RUN cd /src/exatrkx_cpu && ./make.sh -j20 

# Build exatrkx_cpu custom backend
RUN cd /src/custom_backend_cpu/backend && ./make.sh -j20 
RUN cp -r /src/custom_backend_cpu/backend/build/install/install/backends/exatrkxcpu /opt/tritonserver/backends

# Build exatrkx_gpu 
RUN cd /src/exatrkx_gpu && ./make.sh -j20 

# Build exatrkx_gpu custom backend
RUN cd /src/custom_backend_gpu/backend && ./make.sh -j20
RUN cp -r /src/custom_backend_gpu/backend/build/install/install/backends/exatrkxgpu /opt/tritonserver/backends

# Build the client library
RUN cd /src/exatrkx_triton && ./make.sh -j20 

# Copy the model repository files 
RUN mkdir -p /opt/model_repos
RUN cp -r /src/custom_backend_cpu/model_repo/exatrkxcpu /opt/model_repos
RUN cp -r /src/custom_backend_gpu/model_repo/exatrkxgpu /opt/model_repos

# Clean up 
RUN rm -rf /src/* 

