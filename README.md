# Exa.TrkX as a Service

This repository houses the "as-a-service" implementation of the [ExaTrkX](https://arxiv.org/abs/2103.06995) pipeline. We use Nvidia's [Triton inference server](https://github.com/triton-inference-server) to host the ExaTrkX pipeline and schedule requests from clients.

**Figure 1**: ExaTrkX Triton server pipeline

```mermaid
---
title: ExaTrkX ensemble model
---
stateDiagram-v2
    direction LR
    
    classDef pytorch_style fill:#f00,color:white,font-weight:bold,stroke-width:2px,stroke:black
    classDef python_backend_style fill:#46eb34,color:white,font-weight:bold,stroke-width:2px,stroke:yellow
    

    [*] --> embed:::pytorch_style : SP
    embed --> frnn:::python_backend_style : new SP
    [*] --> filter : SP
    frnn --> filter:::pytorch_style : Edges
    filter --> applyfilter:::python_backend_style : Edge Scores
    frnn --> applyfilter : Edges
    applyfilter --> gnn : Edges
    [*] --> gnn:::pytorch_style : SP
    applyfilter --> wcc:::python_backend_style : Edges
    gnn --> wcc : Edge Scores
    wcc --> [*] : Tracks

    state backend_legend {
        direction LR
            pytorch
            python_backend
        }
    

    class pytorch pytorch_style
    class python_backend python_backend_style
```

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

Execute this code during first time setup in order to setup the correct environment + compile ExaTrkx C++ pipeline code:
```bash
source setup.sh
```

Copy the Python Backend tar files into `$SCRATCH/exatrxk/python_backends/`. If you don't have a copy of them you can make them following the instruction in [triton_service/python_backends](triton_service/python_backends/README.md#python-backends).

# Usage 

## Triton Server
Launch the inference server interactively via:
```bash
./deploy_triton.sh
```

Or operate in batch mode via sbatch:
```bash
sbatch --account=<elvis> deploy_triton.sh
```

## Inference Binary
Run the inference binary in parallel with:
```bash
./run_exatrkx.sh server_ip_address:8001 [optional: -i data_folder -n njobs -j cpu_threads_per_job -q/--quiet]
```

## Grafana Dashboard + Nginx Load Balancer
To run the Grafana Dashboard and connect the triton servers to a load balancer run:
```bash
./monitor_triton.sh slurm_jobid
```

To save key metrics run (see [file](triton_service/triton_metrics.py)):
```bash
python triton_service/triton_metrics.py --ip server_ip_address:8002
```

# Notes
