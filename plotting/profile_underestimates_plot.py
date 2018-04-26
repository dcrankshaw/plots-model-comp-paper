import sys
import os
import json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath("../optimizer/"))
import utils
from plotting_utils import get_latencies_inferline_e2e


def load_run(path):
    """
    Loads a single result file and returns utilization and slo miss rate
    """
    with open(path, "r") as f:
        results = json.load(f)
    slo = results["loaded_config"]["slo"]
    lam = results["loaded_config"]["lam"]
    latency_percentage = results["loaded_config"]["latency_percentage"]
    lats = get_latencies_inferline_e2e(results) 
    slo_miss_rate = np.sum(lats > slo) / len(lats)
    return {"latency_percentage_underestimate": 1.0 - latency_percentage, "slo_miss_rate": slo_miss_rate, "lam": lam}

def load_exp(dir_path):

    results = []
    for f in os.listdir(dir_path):
        if f[-4:] == "json":
            # Old result
            if f == "aws_latency_percentage_0.9_lambda_295-180424_185852.json":
                continue
            results.append(load_run(os.path.join(dir_path, f)))
    df = pd.DataFrame(results)
    df = df.sort_values("latency_percentage_underestimate")
    return df

def load_pipeline_three():
    return load_exp(os.path.abspath(
        "../results_cpp_benchmarker/e2e_results/resnet_cascade/prof_underestimate"
        "/util_0.7_with_prune/pipeline_three_prof_underestimate_slo_0.5_cv_1.0_util_0.7"))

def load_pipeline_one():
    return load_exp(os.path.abspath(
        "../results_cpp_benchmarker/e2e_results/image_driver_1/prof_underestimate"
        "/util_0.7/pipeline_one_prof_underestimate_slo_0.5_cv_1.0_util_0.7"))
        
