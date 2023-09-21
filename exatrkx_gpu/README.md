# ExaTrkX in GPUs
This is the preliminary implementation of the ExaTrkX algorithm in GPUs.
The torch models are executed through the `libtorch` library. The fixed
radius clustering is implemented via the `frnn` library. All the dependencies
are installed in a docker image, `docexoty/acts-triton`. You can find the dockerfile
[here](https://github.com/xju2/dockers/blob/main/ML/acts-triton/Dockerfile).

To compile the code `./make.sh` and run the code
```bash
cd build
./bin/inference-gpu -m ../../exatrkx_pipeline/datanmodels -d ../../exatrkx_pipeline/datanmodels/in_e1000.csv
```

Results
```text
Input file: ../../exatrkx_pipeline/datanmodels/in_e1000.csv
Running Inference with local GPUs
Total 37 tracks in 1 events.
1) embedding: 0.0398
2) building:  0.0015
3) filtering: 0.0088
4) gnn:       0.0141
5) labeling:  0.0016
6) total:     0.0659
-----------------------------------------------------
Summary of the first event
1) embedding:  0.0398
2) building:   0.0015
3) filtering:  0.0088
4) gnn:        0.0141
5) labeling:   0.0016
6) total:      0.0659
-----------------------------------------------------
Summary of without first 1 event
Not enough data. 1 total and 1 skipped
Summary of the last event
1) embedding:  0.0398
2) building:   0.0015
3) filtering:  0.0088
4) gnn:        0.0141
5) labeling:   0.0016
6) total:      0.0659
```
