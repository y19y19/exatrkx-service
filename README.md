# Exa.TrkX as a Service

This repository houses the "as-a-service" implementation of the [ExaTrkX](https://arxiv.org/abs/2103.06995) pipeline. We use Nvidia's [Triton inference server](https://github.com/triton-inference-server) to host the ExaTrkX pipeline and schedule requests from clients.

The ExaTrkX pipeline contains 6 stages: Embedding (`embed`), Fixed-radius nearest neighbour (`frnn`), Filtering (`filter`), 
Edge Classification (`gnn`), and Weakly-connected components (`wcc`). 

The pipeline can be run in two modes: direct inference and server inference. We created a [docker file](Dockerfile) that works for both modes.

## Direct Inference
Direct inference means that the algorithm directly runs on CPUs or GPUs without a server. However, we can use the same code in a server and run it on a server.

There are three C++ implementations of the ExaTrkX pipeline: the legacy pipeline that runs on either CPUs or GPUs in [exatrkx_pipeline](exatrkx_pipeline),
the CPU-only pipelin [exatrkx_cpu](exatrkx_cpu), and the GPU-only pipeline [exatrkx_gpu](exatrkx_gpu). Please see instructions in each folder to compile and run the code.

## Triton Server

There are three ExaTrkX-as-a-Service implementations: Ensemble backend in [ensemble_backend](ensemble_backend),
CPU-based customized backend in [custom_backend_cpu](custom_backend_cpu),
and GPU-based customized backend in [custom_backend_gpu](custom_backend_gpu).

## Evaluation

We use the tool `perf_analyzer` [link to Triton doc](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md) from Triton to evaluate the performance. Details can be found in [evaluation](evaluation).
