####################################################
# This script converts the CSV file to a JSON file #
# for the input of perf_analyzer                   #
# Author: Haoran Zhao                              #
# Date: October 2023                               #
####################################################

from pathlib import Path 
import pandas as pd
import numpy as np 
import json 

input_file = Path("../exatrkx_pipeline/datanmodels/in_e1000.csv")

# 1. Read the CSV file into a pandas DataFrame
df = pd.read_csv(input_file, names=['x', 'y', 'z'])

# 2. Convert the DataFrame to a flattened list
flattened_list = df.values.flatten().tolist()

# 3. Create a dictionary with the desired structure
data_structure = {
    "data": [
        {
            "FEATURES": {
                "content": flattened_list,
                "shape": list(df.shape)
            }
        }
    ]
}

with open(input_file.parent / f'{input_file.stem}.json', 'w') as json_file:
    json.dump(data_structure, json_file, indent=4)

