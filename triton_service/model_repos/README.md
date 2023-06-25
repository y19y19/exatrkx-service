# Model Repositories

This folder contains two model repository folders for the nvidia triton inference server to use. It is recommended to use the models contained with the `models` folder. The `models_trackML` was prepared for use with the track ML dataset.

**Figure 1**: ExaTrkX Triton server pipeline

```mermaid
---
title: ExaTrkX ensemble model
---
stateDiagram-v2
    [*] --> embed: exatrkx_input_FEATURES.csv
    embed --> frnn: embed_output_OUTPUT__0.csv
    [*] --> filter
    frnn --> filter: frnn_output_OUTPUT0.csv
    filter --> applyfilter: filter_output_OUTPUT__0.csv
    frnn --> applyfilter
    [*] --> gnn
    applyfilter --> gnn
    applyfilter --> wcc: applyfilter_output_EDGE_LIST_AFTER_FILTER.csv
    gnn --> wcc: gnn_output_OUTPUT__0.csv
    wcc --> [*]
```

## Testing models
Use the `test_triton_model.py` to test models on the triton server:
```bash
python test_triton_model.py [triton_model] [inputs] -t/--triton_address [ip_address]
```

Example:
```bash
python3 test_triton_model.py exatrkx ../../data/exatrkx_input_FEATURES.csv -t 128.55.65.210:8001
```
