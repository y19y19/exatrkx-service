# Build docker customized backend based on tritonserver:22.102-py3

```bash
podman-hpc build -t hrzhao/custom_backend:v0.2 -< docker/Dockerfile
```


```bash 
cd /global/cfs/projectdirs/atlas/hrzhao/ExaTrk/exatrkx-service 

podman-hpc run -it --rm --gpu --shm-size=2g -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ hrzhao/custom_backend:v0.2 
```

## tutorial: recommended backend 
```bash
cd /workspace/custom_backend_cpu
cd examples/backends/recommended/build 

cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BACKEND_REPO_TAG=r22.02 -DTRITON_CORE_REPO_TAG=r22.02 -DTRITON_COMMON_REPO_TAG=r22.02 ../

cp -r install/backends/recommended/ /opt/tritonserver/backends/

cd /workspace/custom_backend_cpu/
tritonserver --model-repository=examples/model_repos/recommended_models/
```

```bash
podman-hpc run -it --rm --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:22.02-py3-sdk bash

cd custom_backend_cpu/examples/
python clients/recommended_client
```