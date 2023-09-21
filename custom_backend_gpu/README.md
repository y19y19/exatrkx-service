# How to run

## start a container
``` bash 
# Plan to push the docker image to dockerhub. But for now, we can build it locally
# podman-hpc pull hrzhao76/custom_backend:2.0 
# Or you can build it locally, and change the image name to $(whoami)/custom_backend_gpu:v0.2 
# It usually takes some time to build the image 
# podman-hpc build -t $(whoami)/custom_backend_gpu:v0.2 -< custom_backend_gpu/docker/Dockerfile
podman-hpc pull hrzhao076/custom_backend:2.0
podman-hpc run -it --rm --gpu --shm-size=2g -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ hrzhao076/custom_backend:2.0 

```


## Compile the backend 
``` bash 
# build the exatrkx_gpu package first, this will install the library to /usr/local/lib
cd /workspace/exatrkx_gpu/ 
./make.sh -j20

cd /workspace/custom_backend_gpu/backend 
./make.sh -j20 

cp -r /workspace/custom_backend_gpu/backend/build/install/install/backends/exatrkxgpu/ /opt/tritonserver/backends && tritonserver --model-repository=/workspace/custom_backend_gpu/model_repo --log-verbose=4
```
## Start the client 

``` bash 
podman-hpc run -it --rm --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/ hrzhao076/custom_backend:2.0 

cd /workspace/exatrkx_triton && ./make.sh -j20 

```

### Check on 100 events
``` bash
time ./build/bin/inference-aas -m exatrkxgpu -d /workspace/exatrkx_pipeline/datanmodels/lrt/inputs
```

### Check on 5000 events 
``` bash 
time ./build/bin/inference-aas -m exatrkxgpu -d /workspace/exatrkx_pipeline/datanmodels/lrt/more
```
