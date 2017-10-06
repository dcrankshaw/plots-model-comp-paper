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
                if len(data["client_metrics"]["thrus"]) > 5:
                    experiments.append(data)
        else:
            print("skipping %s" % os.path.join(results_dir, exp))
    return experiments


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
    # discard first trial
    thrus = thrus[1:]
    mean = np.mean(thrus)
    std = np.std(thrus)
    return (mean, std)


def client_lat(exp):
    lats = exp["client_metrics"]["mean_lats"]
    # discard first trial
    return np.mean(lats[1:])


def extract_client_metrics(exp):
    name = model_name(exp)
    hists = exp["clipper_metrics"]["histograms"]
    lat_key = "model:{name}:1:prediction_latency".format(name=name)
    batch_key = "model:{name}:1:batch_size".format(name=name)
    for h in hists:
        if list(h.keys())[0] == lat_key:
            p99_lat = h[lat_key]["p99"]
        elif list(h.keys())[0] == batch_key:
            mean_batch = h[batch_key]["mean"]
    return (p99_lat, mean_batch)


def create_results_df(results_dir, expected_model_name):
    experiments = load_results(results_dir)

    gpus = []
    cpus = []
    config_batch = []
    # model_name = []
    mean_thru = []
    std_thru = []
    p99_lat = []
    mean_batch = []
    client_lats = []

    for e in experiments:
        if model_name(e) == expected_model_name:
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
        else:
            print("Found experiment with model: %s" % model_name(e))

    results_dict = {
        "num_gpus": gpus,
        "num_cpus": cpus,
        "batch_size": config_batch,
        "mean_throughput": mean_thru,
        "std_throughput": std_thru,
        "p99_latency": p99_lat,
        "actual_batch": mean_batch,
        "client_latency": client_lats
    }

    df = pd.DataFrame.from_dict(results_dict)
    return df


if __name__ == '__main__':

    models = [
        ("tf_lstm", "lstm"),
        ("lgbm", "lgbm"),
        ("tf_inception", "inception"),
    ]

    for d, m in models:
        print(m)
        results_dir = os.path.join(os.path.abspath("../results/single_model_profs/"), d)
        df = create_results_df(results_dir, m)
        df.to_csv(os.path.join(results_dir, "summary.csv"))






