import sys
import os
import json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath("../optimizer/"))
import utils

def get_latencies_inferline_e2e(results_json):
    trials = results_json["throughput_results"]["client_metrics"][0][1:]
    lats = []
    for t in trials:
        datalists = t["data_lists"]
        for d in datalists:
            if list(d.keys())[0] == "e2e:prediction_latencies":
                items = d["e2e:prediction_latencies"]["items"]
                for i in items:
                    lats.append(float(list(i.values())[0]) / 1000.0 / 1000.0)
    lats = np.array(lats)
    return lats
