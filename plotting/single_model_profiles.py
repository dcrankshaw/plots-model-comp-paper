import os
# import sys
import numpy as np
# import matplotlib.pyplot as plt
import json
# import seaborn as sns
import pandas as pd
import itertools
from tabulate import tabulate

from utils import COST_PER_GPU, COST_PER_CPU

def load_results(results_dir):
    fs = os.listdir(results_dir)
    experiments = []
    for exp in fs:
        if exp[-4:] == "json":
            with open(os.path.join(results_dir, exp), "r") as f:
                data = json.load(f)
                format_client_metrics(data)
                if max([len(cm["thrus"]) for cm in data["client_metrics"]]) > 5:
                    experiments.append(data)
        else:
            # print("skipping %s" % os.path.join(results_dir, exp))
            pass
    return experiments


def format_client_metrics(data):
    if type(data["client_metrics"]) == dict:
        data["client_metrics"] = [data["client_metrics"]]


def num_gpus(exp):
    return len(exp["node_configs"][0]["gpus"])


def num_cpus(exp):
    return exp["node_configs"][0]["cpus_per_replica"]


def batch_size(exp):
    return exp["node_configs"][0]["batch_size"]


def model_name(exp):
    return exp["node_configs"][0]["name"]


def instance_type(exp):
    if "instance_type" in exp["node_configs"][0]:
        return exp["node_configs"][0]["instance_type"]
    else:
        return "g3.4xlarge"


def throughput(exp):
    mean = 0
    var = 0
    for cm in exp["client_metrics"]:
        # discard first trial for each set of client metrics
        cm_thrus = cm["thrus"][1:]
        cm_mean = np.mean(cm_thrus)
        cm_var = np.var(cm_thrus)
        mean += cm_mean
        var += cm_var

    std = np.sqrt(var)
    return (mean, std)


def client_lat(exp):
    # discard first trial for each set of client metrics
    all_lats = [cm["mean_lats"][1:] for cm in exp["client_metrics"]]
    all_lats = list(itertools.chain.from_iterable(all_lats))
    return np.mean(all_lats) * 1000.0


def extract_client_metrics(exp):
    name = model_name(exp)
    hists = exp["clipper_metrics"]["histograms"]
    lat_key = "model:{name}:1:prediction_latency".format(name=name)
    batch_key = "model:{name}:1:batch_size".format(name=name)
    for h in hists:
        if list(h.keys())[0] == lat_key:
            p99_lat = float(h[lat_key]["p99"]) / 1000.0
        elif list(h.keys())[0] == batch_key:
            mean_batch = h[batch_key]["mean"]
    return (p99_lat, mean_batch)


def compute_cost(results_json):
    nodes = results_json["node_configs"]
    total_cost = 0.0
    for n in nodes:
        num_reps = n["num_replicas"]
        n_gpus = num_gpus(results_json) * num_reps
        n_cpus = num_cpus(results_json) * num_reps
        cost = float(n_gpus) * COST_PER_GPU + float(n_cpus) * COST_PER_CPU
        total_cost += cost
    return total_cost


def create_model_profile_df_old_format(results_dir):
    experiments = load_results(results_dir)

    gpus = []
    cpus = []
    config_batch = []
    model_names = []
    mean_thru = []
    std_thru = []
    p99_lat = []
    mean_batch = []
    client_lats = []
    inst_types = []
    costs = []

    for e in experiments:
        model_names.append(model_name(e))
        inst_types.append(instance_type(e))
        gpus.append(num_gpus(e))
        cpus.append(num_cpus(e))
        config_batch.append(batch_size(e))
        mean_t, std_t = throughput(e)
        mean_thru.append(mean_t)
        std_thru.append(std_t)
        costs.append(compute_cost(e))
        p99_l, mean_b = extract_client_metrics(e)
        p99_lat.append(p99_l)
        mean_batch.append(mean_b)
        client_lats.append(client_lat(e))

    results_dict = {
        "num_gpus_per_replica": gpus,
        "num_cpus_per_replica": cpus,
        "mean_throughput_qps": mean_thru,
        "std_throughput_qps": std_thru,
        "p99_latency_ms": p99_lat,
        "mean_batch_size": mean_batch,
        "client_latency_ms": client_lats,
        "inst_type": inst_types,
        "cost": costs
    }

    df = pd.DataFrame.from_dict(results_dict)
    return df

# if __name__ == '__main__':
#
#     single_model_profs_dir = os.path.abspath("../results/single_model_profs/")
#     for m in os.listdir(single_model_profs_dir):
#         print(m)
#         results_dir = os.path.join(single_model_profs_dir, m)
#         df = create_results_df(results_dir)
#         df.to_csv(os.path.join(results_dir, "summary.csv"))
#         if "pytorch" in m:
#             with open(os.path.join(results_dir, "summary_pretty.tab"), "w") as f:
#                 df_summary = df.loc[(df["inst_type"] == "p2.8xlarge") & (df["num_cpus_per_replica"] == 1) & (df["num_gpus_per_replica"] == 1)]
#                 df_summary = df_summary.filter(items=["configured_batch_size",
#                                                       "mean_throughput_qps",
#                                                       "p99_latency_ms"])
#
#                 f.write(tabulate(df_summary,
#                                  headers='keys',
#                                  tablefmt='psql'))
