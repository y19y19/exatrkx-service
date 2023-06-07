# Exa.TrkX as a Service

This repository houses the "as-a-service" implementation of the [ExaTrkX](https://arxiv.org/abs/2103.06995) pipeline. We use Nvidia's [Triton inference server](https://github.com/triton-inference-server) to host the ExaTrkX pipeline and schedule requests from clients.

**Table 1**: ExaTrkX Triton server pipeline
**Stage**|**Backend**
:-----|:-----
`embed`| PyTorch
`frnn`| Python
`filter`| PyTorch
`applyfilter`| Python
`gnn`| PyTorch
`wcc`| Python


# Setup

Execute this code during first time setup in order to setup the correct environment:
```bash
source setup.sh
```

Copy the Python Backend tar files into `$SCRATCH/exatrxk/python_backends/`. If you don't have a copy of them you can make them following the instruction in [Python Backends](python_backends/README.md#python-backends).

Once complete launch the inference server via:
```bash
source deploy_triton.sh
```

# Usage 


# Notes