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
from prometheus_client.parser import text_string_to_metric_families #pip install prometheus_client
from time import sleep
from copy import deepcopy
import numpy as np


#Config
poll_freq = 1 #1s
ntimes = 20 #20
timeout = poll_freq * ntimes
models = ['exatrkx', 'embed', 'gnn', 'filter', 'applyfilter', 'wcc', 'frnn']

delta_func = lambda x: x[-1] - x[0]
diff_func = lambda x: f'{delta_func(x):.0f}'
avg_func = lambda x, y: f'{(delta_func(x)/(1000)/delta_func(y)):.4f} ms per event'

interested_metrics = {
    'nv_inference_request_success': {'name': 'Success Count', 'func': diff_func},
    'nv_inference_request_duration_us': {'name': 'Request Time', 'func': avg_func},
    'nv_inference_queue_duration_us': {'name': 'Queue Time', 'func': avg_func},
    'nv_inference_compute_input_duration_us': {'name': 'Compute Input Time', 'func': avg_func},
    'nv_inference_compute_infer_duration_us': {'name': 'Compute Time', 'func': avg_func},
    'nv_inference_compute_output_duration_us': {'name': 'Compute Output Time', 'func': avg_func}
}

metrics_labels = interested_metrics.keys()
models_empty_dict = {k:[] for k in models}
metrics_data = {k:deepcopy(models_empty_dict) for k in interested_metrics}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        triton_metrics_ip = sys.argv[1]
    else:
        triton_metrics_ip = 'localhost:8002'

    triton_metrics_path = f'http://{triton_metrics_ip}/metrics'

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
        print(f'<> {interested_metrics[k]["name"]}')
        for m in models:
            if k == 'nv_inference_request_success':
                print(f'    - {m} - {interested_metrics[k]["func"](v[m])}')
            else:
                print(f'    - {m} - {interested_metrics[k]["func"](v[m], metrics_data["nv_inference_request_success"][m])}')
