import numpy as np
import json
import os
import sys
import pandas as pd

from utils import COST_PER_GPU, COST_PER_CPU


def load_results(results_dir):
    fs = os.listdir(results_dir)
    experiments = []
    for exp in fs:
        if exp[-4:] == "json":
            with open(os.path.join(results_dir, exp), "r") as f:
                data = json.load(f)
                # Skip the first trial
                skip_first_trial(data)
                experiments.append(data)
        else:
            # print("skipping %s" % os.path.join(results_dir, exp))
            pass
    return experiments

def skip_first_trial(results_json):
    client_metrics = results_json["client_metrics"]
    for client in client_metrics:
        for metric in client:
            client[metric] = client[metric][1:]

# # heuristic to determine when latencies have flattened out
# def select_valid_trials(name, results_json):
#     p99_lats = results_json["client_metrics"][0]["p99_lats"]
#
#     # We assume that at least the last 8 trials were good
#     last_8_mean = np.mean(p99_lats[-8:])
#     last_8_stdev = np.mean(p99_lats[-8:])
#
#     good_trials = []
#     for i in reversed(range(len(p99_lats))):
#         if p99_lats[i] <= last_8_mean + last_8_stdev:
#             good_trials.append(i)
#         elif len(p99_lats) - i < 8:
#             print("Found a bad trial in the last 8 trials for: {}".format(name))
#         else:
#             break
#     first_good_trial = min(good_trials)
#     last_good_trial = max(good_trials)
#     assert last_good_trial == len(p99_lats) - 1
#     return first_good_trial, last_good_trial


# def extract_good_results(results_json, first_good_trial, last_good_trial):
#     client_metrics = results_json["client_metrics"]
#     for client in client_metrics:
#         for metric in client:
#             client[metric] = client[metric][first_good_trial:last_good_trial + 1]

def num_gpus(exp):
    return exp["node_configs"][0]["gpus_per_replica"]


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


def extract_client_metrics(exp):
    clipper_metrics = exp["client_metrics"][0]["all_metrics"]
    p99_lats = []
    mean_batch_sizes = []
    name = model_name(exp)

    for trial in clipper_metrics:
        hists = trial["histograms"]
        lat_key = "model:{name}:1:prediction_latency".format(name=name)
        batch_key = "model:{name}:1:batch_size".format(name=name)
        for h in hists:
            if list(h.keys())[0] == lat_key:
                lat = float(h[lat_key]["p99"]) / 1000.0
                # if float(h[lat_key]["p99"]) / 1000.0 < 0.01:
                #     # print(json.dumps(trial, indent=2))
                #     continue
                # else:
                p99_lats.append(lat)
            elif list(h.keys())[0] == batch_key:
                mean_batch_sizes.append(float(h[batch_key]["mean"]))
    return (np.mean(p99_lats), np.std(p99_lats), np.mean(mean_batch_sizes))

def compute_cost(results_json):
    nodes = results_json["node_configs"]
    total_cost = 0.0
    for n in nodes:
        num_reps = n["num_replicas"]
        num_gpus = n["gpus_per_replica"] * num_reps
        num_cpus = n["cpus_per_replica"] * num_reps
        cost = float(num_gpus) * COST_PER_GPU + float(num_cpus) * COST_PER_CPU
        total_cost += cost
    return total_cost

def create_model_profile_df(results_dir):
    experiments = load_results(results_dir)

    gpus = []
    cpus = []
    mean_thrus = []
    std_thrus = []
    p99_lats = []
    p99_lat_errs = []
    mean_batches = []
    inst_types = []
    costs = []

    for e in experiments:
        inst_types.append(instance_type(e))
        gpus.append(num_gpus(e))
        cpus.append(num_cpus(e))
        mean_thru, std_thru = throughput(e)
        mean_thrus.append(mean_thru)
        std_thrus.append(std_thru)
        costs.append(compute_cost(e))
        p99_lat, p99_lat_err, mean_batch = extract_client_metrics(e)
        p99_lats.append(p99_lat)
        p99_lat_errs.append(p99_lat_err)
        mean_batches.append(mean_batch)

    results_dict = {
        "num_gpus_per_replica": gpus,
        "num_cpus_per_replica": cpus,
        "mean_throughput_qps": mean_thrus,
        "std_throughput_qps": std_thrus,
        "p99_latency_ms": p99_lats,
        "p99_latency_ms_stddev": p99_lat_errs,
        "mean_batch_size": mean_batches,
        "inst_type": inst_types,
        "cost": costs
    }

    df = pd.DataFrame.from_dict(results_dict)
    df = df[list(results_dict.keys())]
    return df


