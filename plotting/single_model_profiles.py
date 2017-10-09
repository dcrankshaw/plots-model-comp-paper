import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd


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
            print("skipping %s" % os.path.join(results_dir, exp))
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


def throughput(exp):
    thrus = exp["client_metrics"]["thrus"]
    mean = 0
    var = 0
    for cm in exp["client_metrics"]:
        # discard first trial
        cm_thrus = cm["thrus"][1:]
        cm_mean = np.mean(cm_thrus)
        cm_var = np.var(cm_thrus)
        mean += cm_mean
        var += cm_var

    std = np.sqrt(var)
    return (mean, std)

def client_lat(exp):
    lats = exp["client_metrics"]["mean_lats"]
    # discard first trial
    all_lats = np.array([cm["mean_lats"][1:] for cm in exp["client_metrics"]]).flatten()
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


def create_results_df(results_dir):
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

    for e in experiments:
        model_names.append(model_name(e))
        gpus.append(num_gpus(e))
        cpus.append(num_cpus(e))
        config_batch.append(batch_size(e))
        mean_t, std_t = throughput(e)
        mean_thru.append(mean_t)
        std_thru.append(std_t)

        p99_l, mean_b = extract_client_metrics(e)
        p99_lat.append(p99_l)
        mean_batch.append(mean_b)
        client_lats.append(client_lat(e))

    results_dict = {
        "model_name" : model_names,
        "num_gpus_per_replica": gpus,
        "num_cpus_per_replica": cpus,
        "configured_batch_size": config_batch,
        "mean_throughput_qps": mean_thru,
        "std_throughput_qps": std_thru,
        "p99_latency_ms": p99_lat,
        "mean_batch_size": mean_batch,
        "client_latency_ms": client_lats
    }

    df = pd.DataFrame.from_dict(results_dict)
    return df


if __name__ == '__main__':

    single_model_profs_dir = os.path.abspath("../results/single_model_profs/")
    for m in os.listdir(single_model_profs_dir):
        print(m)
        results_dir = os.path.join(single_model_profs_dir, m)
        df = create_results_df(results_dir)
        df.to_csv(os.path.join(results_dir, "summary.csv"))






