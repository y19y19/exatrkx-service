# Build docker customized backend based on tritonserver:22.02-py3

## How to Run

### Build a container
As an example in Perlmutter, we use `podman-hpc` to build and run the container.

First build the container in a login node. We use `user-name` as an example.
```bash
podman-hpc build -t your-username/custom_backend:v0.6 -< docker/Dockerfile
```
And then `migrate` the image to use it in a job. 
```
podman-hpc migrate user-name/custom_backend:v0.6
```
After that, a read-only image is created.

### Set up the server 
Run 
```
podman-hpc run -it --rm --gpu --shm-size=2g -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ user-name/custom_backend:v0.6
```

* compile the `exatrkx_cpu` by running `make.sh` inside `exatrkx_cpu`.
* compile the customized backend by running the `make.sh` inside `custom_backend_cpu/backend`

Launch the server:

```bash
cp -r /workspace/custom_backend_cpu/backend/build/install/install/backends/exatrkxcpu /opt/tritonserver/backends && tritonserver --model-repository=/workspace/custom_backend_cpu/model_repo --log-verbose=4
```

### Start the client

Run 
```
podman-hpc run -it --rm --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/ user-name/custom_backend:v0.6
```
Then `cd /workspace/exatrkx_triton && ./make.sh`.

Check on 100 events
```bash!
time ./build/bin/inference-aas -d /workspace/exatrkx_pipeline/datanmodels/lrt/inputs
```
