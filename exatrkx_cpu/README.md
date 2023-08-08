# ExaTrkX in CPUs
This is the preliminary implementation of the ExaTrkX algorithm in CPUs.
The torch models are executed through the `libtorch` library. The fixed
radius clustering is implemented via the `faiss-cpu` library. All the dependencies
are installed in a docker image, `docexoty/exatrkx-cpu`. You can find the dockerfile
[here](https://github.com/xju2/dockers/blob/main/ML/exatrkx-cpu/Dockerfile).

To compile the code `./make.sh` and run the code
```bash
cd build
./bin/inference-cpu -m ../../exatrkx_pipeline/datanmodels -d ../../exatrkx_pipeline/datanmodels/in_e1000.csv
```

Results
```text
Input file: ../../exatrkx_pipeline/datanmodels/in_e1000.csv
Models loaded successfully
Running Inference with local CPUs
Embedding model run successfully
is_trained = true
Total 39 tracks in 1 events.
1) embedding: 0.4282
2) building:  0.0166
3) filtering: 3.4804
4) gnn:       0.2146
5) labeling:  0.0023
6) total:     4.1421
-----------------------------------------------------
Summary of the first event
1) embedding:  0.4282
2) building:   0.0166
3) filtering:  3.4804
4) gnn:        0.2146
5) labeling:   0.0023
6) total:      4.1421
-----------------------------------------------------
Summary of without first 1 event
Not enough data. 1 total and 1 skipped
Summary of the last event
1) embedding:  0.4282
2) building:   0.0166
3) filtering:  3.4804
4) gnn:        0.2146
5) labeling:   0.0023
6) total:      4.1421
```
