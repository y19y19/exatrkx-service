# Build docker customized backend based on tritonserver:22.02-py3

```bash
podman-hpc build -t hrzhao/custom_backend:v0.3 -< docker/Dockerfile
```


```bash 
cd /global/cfs/projectdirs/atlas/hrzhao/ExaTrk/exatrkx-service 

podman-hpc run -it --rm --gpu --shm-size=2g -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ hrzhao/custom_backend:v0.5

cd /workspace/custom_backend_cpu/backend_1/build
rm -rf * && cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ../  -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" && make install -j20 

```

```bash
./bin/inference-cpu -m ../../../exatrkx_pipeline/datanmodels -d ../../../exatrkx_pipeline/datanmodels/in_e1000.csv

cp -r install/backends/exatrkxcpu/ /opt/tritonserver/backends
tritonserver --model-repository=/workspace/custom_backend_cpu/model_repo/

```

Useful commands:
``` bash
rm -rf * && cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ../ && make install -j20

rm -rf * && cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ../  -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" && make install -j20  && cp -r install/backends/exatrkxcpu/ /opt/tritonserver/backends && tritonserver --model-repository=/workspace/custom_backend_cpu/model_repo/ --log-verbose=4

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
podman-hpc run -it --rm --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.07-py3-sdk bash

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

## ldd

```
        linux-vdso.so.1 (0x00007ffca4b60000)                                                                                                                                                                                               
        libtritonserver.so => not found                                                                                                                                                                                                    
        libtorch_cpu.so => not found                                                                                                                                                                                                       
        libc10.so => not found                                                                                                                                                                                                             
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f48c8b2e000)                                                                                                                                                      
        libtorch.so => not found                                                                                                                                                                                                           
        libgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007f48c8aea000)                                                                                                                                                            
        libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f48c8908000)                                                                                                                                                        
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f48c88ed000)                                                                                                                                                          
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f48c86fb000)                                                                                                                                                                  
        /lib64/ld-linux-x86-64.so.2 (0x00007f48c8bad000)                                                                                                                                                                                   
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f48c86f5000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f48c85a6000)
        ```



```
root@ffd5ef261239:/workspace/custom_backend_cpu/backend/build# ldd ../../../exatrkx_cpu/build/lib/libExaTrkXCPU.so 
        linux-vdso.so.1 (0x00007ffc02ac5000)
        libc10.so => /usr/local/lib/python3.8/dist-packages/torch/lib/libc10.so (0x00007f531d483000)
        libtorch_cpu.so => /usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cpu.so (0x00007f530689a000)
        libtorch.so => /usr/local/lib/python3.8/dist-packages/torch/lib/libtorch.so (0x00007f5306897000)
        libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f53066ab000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f530655c000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f530653f000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f530634d000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f531d655000)
        libgomp-a34b3233.so.1 => /usr/local/lib/python3.8/dist-packages/torch/lib/libgomp-a34b3233.so.1 (0x00007f5306123000)
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f5306100000)
        librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007f53060f5000)
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f53060ef000)

```


```
/opt/tritonserver/backends/exatrkxcpu/libtriton_exatrkxcpu.so: undefined symbol: _ZN6triton7backend13ShapeToStringERKSt6vectorIlSaIlEE

nm libtriton_exatrkxcpu.so | c++filt | grep ShapeToString
```


```
root@3457f49d241b:/workspace/custom_backend_cpu/backend_backup/build# nm libtriton_exatrkxcpu.so | c++filt | grep ShapeToString
0000000000033810 t triton::backend::ShapeToString[abi:cxx11](long const*, unsigned long)
00000000000098f9 t triton::backend::ShapeToString[abi:cxx11](long const*, unsigned long) [clone .cold]
0000000000033980 t triton::backend::ShapeToString[abi:cxx11](std::vector<long, std::allocator<long> > const&)
                 U triton::backend::ShapeToString(std::vector<long, std::allocator<long> > const&)

python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```

```
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```