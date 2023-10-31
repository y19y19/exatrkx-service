# Acts Demostrator 

This folder contains the models from Benjamin's repo [exatrkx-acts-demonstrator](https://github.com/benjaminhuth/exatrkx-acts-demonstrator) and the `model_repo` for the triton server backend config. 


``` bash 
podman-hpc run -it --rm --gpu --shm-size=2g -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ hrzhao076/exatrkx_triton_backend:4.0 bash
```

## Direct inference 
``` bash 
cd /workspace/exatrkx_gpu/ 
./make.sh -j20 

inference-gpu -m /workspace/exatrkx_pipeline/datanmodels/ -d /workspace/exatrkx_pipeline/datanmodels/in_e1000.csv

inference-gpu -m /workspace/acts_demostrator/models/true_hits/ -d /workspace/exatrkx_pipeline/datanmodels/in_e1000.csv

inference-gpu -e acts-smear -m /workspace/acts_demostrator/models/smeared_hits/ -d /workspace/exatrkx_pipeline/datanmodels/in_e1000.csv

```


## Inference with triton server 

``` bash 
# compile the `exatrkx_gpu` first
cd /workspace/exatrkx_gpu/ 
./make.sh -j20 

# start the triton server in the background 
nohup tritonserver --model-repository=/workspace/acts_demostrator/model_repo/ > log.test.txt 2>&1 &

# send the request to the triton server
inference-aas -m exatrkxgpu-acts-truehits -d /workspace/exatrkx_pipeline/datanmodels/in_e1000.csv
inference-aas -m exatrkxgpu-acts-smearhits -d /workspace/exatrkx_pipeline/datanmodels/in_e1000.csv

```

An example output is shown below: 


```
# test the truth hits 
root@1c8336f44b6f:/workspace/custom_backend_gpu/backend# inference-aas -m exatrkxgpu-acts-truehits -d /workspace/exatrkx_pipeline/datanmodels/in_e1000.csv
Input file: /workspace/exatrkx_pipeline/datanmodels/in_e1000.csv
Running Inference with ExaTrkX as a service.
Total 256 tracks in 1 events

# test the smeared hits 
root@1c8336f44b6f:/workspace/custom_backend_gpu/backend# inference-aas -m exatrkxgpu-acts-smearhits -d /workspace/exatrkx_pipeline/datanmodels/in_e1000.csv
Input file: /workspace/exatrkx_pipeline/datanmodels/in_e1000.csv
Running Inference with ExaTrkX as a service.
Total 113 tracks in 1 events
```