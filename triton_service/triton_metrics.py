#!/usr/bin/env python

#=========================================
# 
# Title: triton_metrics.py
# Author: Andrew Naylor
# Date: Jun 23
# Brief: Print out metrics to terminal
#
# Usage: python triton_metrics.py server_address
#
#=========================================

import sys
import requests
from prometheus_client.parser import text_string_to_metric_families
from time import sleep
from copy import deepcopy
import numpy as np

models = ['exatrkx', 'embed', 'gnn', 'filter', 'applyfilter', 'wcc', 'frnn']
models_empty_dict = {k:[] for k in models}

interested_metrics = {
    'nv_inference_request_success': 'Success Count',
    'nv_inference_request_duration_us': 'Request Time',
    'nv_inference_queue_duration_us': 'Queue Time',
    'nv_inference_compute_input_duration_us': 'Compute Input Time',
    'nv_inference_compute_infer_duration_us': 'Compute Time',
    'nv_inference_compute_output_duration_us': 'Compute Output Time'}
metrics_labels = interested_metrics.keys()

metrics_data = {k:deepcopy(models_empty_dict) for k in interested_metrics}

poll_freq = 1 #1s
ntimes = 20
timeout = poll_freq * ntimes


if __name__ == "__main__":
    if len(sys.argv) > 1:
        triton_metrics_ip = sys.argv[1]
    else:
        triton_metrics_ip = 'localhost'

    triton_metrics_path = f'http://{triton_metrics_ip}:8002/metrics'

    #Loop until timeout
    print("<pulling data>")
    timer = 0
    while timer < timeout:
        res = requests.get(triton_metrics_path)
        res_data = res.content.decode('utf-8')
        _interested_metrics = interested_metrics.copy()

        for i in text_string_to_metric_families(res_data):
            if i.name in metrics_labels:
                # print(f'<> Processing metric {i.name}')
                for j in i.samples:
                    # print(f"{j.labels['model']} - {j.value}")
                    metrics_data[i.name][j.labels['model']].append(j.value)
                
                _interested_metrics.pop(i.name)
    
            if len(_interested_metrics) == 0:
                break
        sleep(poll_freq)
        timer+=poll_freq
    
    #Print out metrics
    for k, v in metrics_data.items():
        print(f'<> {interested_metrics[k]}')
        for m in models:
            if k != 'nv_inference_request_success':
                print(f'    - {m} - {np.average(np.diff(v[m])/1000)} ms')
            else:
                print(f'    - {m} - {np.ptp(v[m])}')

