# Build docker customized backend based on tritonserver:22.02-py3

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

make install -j8 

cp -r install/backends/recommended/ /opt/tritonserver/backends/

cd /workspace/custom_backend_cpu/
tritonserver --model-repository=examples/model_repos/recommended_models/
```

```bash
podman-hpc run -it --rm --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:22.02-py3-sdk bash

cd custom_backend_cpu/examples/
python clients/recommended_client
```

## Example backends 
[PyTorch Backend](https://github.com/triton-inference-server/pytorch_backend) 
[Python Backend](https://github.com/triton-inference-server/python_backend)



# Dev 

## long unsigned int vs int 
```
/workspace/custom_backend_cpu/src/exatrkx_cpu/ExaTrkXTiming.hpp: In member function 'ExaTrkXTime ExaTrkXTimeList:
:get(int)':                                                                                                      
/workspace/custom_backend_cpu/src/exatrkx_cpu/ExaTrkXTiming.hpp:45:19: error: comparison of integer expressions o
f different signedness: 'int' and 'std::vector<float, std::allocator<float> >::size_type' {aka 'long unsigned int
'} [-Werror=sign-compare]                                                                                        
   45 |         if (evtid >= embedding.size()) {                                                                 
      |             ~~~~~~^~~~~~~~~~~~~~~~~~~                                                                    
/workspace/custom_backend_cpu/src/exatrkx_cpu/ExaTrkXTiming.hpp: In member function 'void ExaTrkXTimeList::summar
y(int)':                                                                                                         
/workspace/custom_backend_cpu/src/exatrkx_cpu/ExaTrkXTiming.hpp:62:17: error: comparison of integer expressions o
f different signedness: 'size_t' {aka 'long unsigned int'} and 'int' [-Werror=sign-compare]
   62 |         if (num <= start) {
      |             ~~~~^~~~~~~~
/workspace/custom_backend_cpu/src/exatrkx_cpu/ExaTrkXTiming.hpp: In member function 'void ExaTrkXTimeList::save(c
onst string&)':
/workspace/custom_backend_cpu/src/exatrkx_cpu/ExaTrkXTiming.hpp:105:27: error: comparison of integer expressions 
of different signedness: 'int' and 'std::vector<float, std::allocator<float> >::size_type' {aka 'long unsigned in
t'} [-Werror=sign-compare]
  105 |         for (int i = 0; i < embedding.size(); i++) {

```
Solved by: `int num = static_cast<int>(embedding.size());`

## Naming Convention 
`ExatrkxCPU` and `ExatrkXCPU`