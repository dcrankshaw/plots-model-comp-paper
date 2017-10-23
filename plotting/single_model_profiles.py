import numpy as np
import json
import os
# import sys
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
                format_client_metrics(data)
                skip_first_trial(data)
                if max([len(cm["thrus"]) for cm in data["client_metrics"]]) > 5:
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
        cm_thrus = cm["thrus"]
        cm_mean = np.mean(cm_thrus)
        cm_var = np.var(cm_thrus)
        mean += cm_mean
        var += cm_var

    std = np.sqrt(var)
    return (mean, std)


def extract_client_metrics_old(exp):
    name = model_name(exp)
    hists = exp["clipper_metrics"]["histograms"]
    lat_key = "model:{name}:1:prediction_latency".format(name=name)
    batch_key = "model:{name}:1:batch_size".format(name=name)
    for h in hists:
        if list(h.keys())[0] == lat_key:
            p99_lat = float(h[lat_key]["p99"]) / 1000.0
        elif list(h.keys())[0] == batch_key:
            mean_batch = float(h[batch_key]["mean"])
    return (p99_lat, mean_batch)


def extract_client_metrics_new(exp):
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
                p99_lats.append(lat)
            elif list(h.keys())[0] == batch_key:
                mean_batch_sizes.append(float(h[batch_key]["mean"]))
    return (np.mean(p99_lats), np.mean(mean_batch_sizes))


def extract_client_metrics(exp):
    if "clipper_metrics" in exp:
        return extract_client_metrics_old(exp)
    else:
        return extract_client_metrics_new(exp)


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


def create_model_profile_df(results_dir):
    experiments = load_results(results_dir)

    gpus = []
    cpus = []
    mean_thrus = []
    std_thrus = []
    p99_lats = []
    mean_batches = []
    inst_types = []
    costs = []

    model_name = experiments[0]["node_configs"][0]["name"]

    for e in experiments:
        inst_types.append(instance_type(e))
        gpus.append(num_gpus(e))
        cpus.append(num_cpus(e))
        mean_thru, std_thru = throughput(e)
        mean_thrus.append(mean_thru)
        std_thrus.append(std_thru)
        costs.append(compute_cost(e))
        p99_lat, mean_batch = extract_client_metrics(e)
        p99_lats.append(p99_lat)
        mean_batches.append(mean_batch)

    results_dict = {
        "num_gpus_per_replica": gpus,
        "num_cpus_per_replica": cpus,
        "mean_throughput_qps": mean_thrus,
        "std_throughput_qps": std_thrus,
        "p99_latency_ms": p99_lats,
        "mean_batch_size": mean_batches,
        "inst_type": inst_types,
        "cost": costs
    }

    df = pd.DataFrame.from_dict(results_dict)
    df = df[list(results_dict.keys())]
    return (model_name, df)


def load_single_model_profiles(
        single_model_profs_dir=os.path.abspath("../results/single_model_profs/")):
    """
    Returns
    -------
    dict :
        A dict that maps model name to a DataFrame containing single model profile for that
        model. The model name is set based on the node config in the experimental results. It will
        not necessarily match the directory name of the directory containing the profiling results
        for that model.
    """
    profs = {}
    for m in os.listdir(single_model_profs_dir):
        fname = os.path.join(single_model_profs_dir, m)
        if os.path.isdir(fname):
            model_name, df = create_model_profile_df(fname)
            profs[model_name] = df
    return profs
