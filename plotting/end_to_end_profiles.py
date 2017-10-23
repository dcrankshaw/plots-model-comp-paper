import numpy as np
import json
import os
# import sys
import pandas as pd

COST_PER_CPU = 4.75 / 100.0
COST_PER_GPU = 70.0 / 100.0


def load_end_to_end_results(results_dir):
    fs = os.listdir(results_dir)
    experiments = {}
    for exp in fs:
        if exp[-4:] == "json":
            with open(os.path.join(results_dir, exp), "r") as f:
                data = json.load(f)
                name = exp[:-5]
                first_good_trial, last_good_trial = select_valid_trials(name, data)
                print(first_good_trial, last_good_trial)
                extract_good_results(data, first_good_trial, last_good_trial)
                experiments[name] = data
        else:
            # print("skipping %s" % os.path.join(results_dir, exp))
            pass
    return experiments


# heuristic to determine when latencies have flattened out
def select_valid_trials(name, results_json):
    p99_lats = results_json["client_metrics"][0]["p99_lats"]

    num_good_trials = 8

    # We assume that at least the last 8 trials were good
    last_8_mean = np.mean(p99_lats[-1*num_good_trials:])
    last_8_stdev = np.std(p99_lats[-1*num_good_trials:])

    # good_trials = []
    # for i in reversed(range(len(p99_lats))):
    #     if p99_lats[i] <= last_8_mean + last_8_stdev:
    #         good_trials.append(i)
    #     elif len(p99_lats) - i < num_good_trials:
    #         # print("Found a bad trial in the last 8 trials for: {}".format(name))
    #         continue
    #     else:
    #         break
    # first_good_trial = min(good_trials)
    # last_good_trial = max(good_trials)
    # assert len(good_trials) > 1
    # assert last_good_trial == len(p99_lats) - 1
    # return first_good_trial, last_good_trial
    return len(p99_lats) - num_good_trials - 1, len(p99_lats) - 1


def extract_good_results(results_json, first_good_trial, last_good_trial):
    node_configs = results_json["node_configs"]
    results_json["node_configs"] = [n for n in node_configs if "request_delay" not in n]
    client_metrics = results_json["client_metrics"]
    for client in client_metrics:
        # First deal with inconsistently formatted all_lats list:
        lat_entries_per_trial = len(client["all_lats"]) / len(client["p99_lats"])
        if lat_entries_per_trial > 1:
            print("Found {} queries per trial".format(lat_entries_per_trial))
            first_entry = round(first_good_trial * lat_entries_per_trial)
            last_entry = round((last_good_trial + 1) * lat_entries_per_trial)
            client["all_lats"] = client["all_lats"][first_entry:last_entry]
        else:
            client["all_lats"] = client["all_lats"][first_good_trial:last_good_trial + 1]
        for metric in client:
            if metric == "all_lats":
                continue
            else:
                client[metric] = client[metric][first_good_trial:last_good_trial + 1]


def get_throughput(results_json):
    client_metrics = results_json["client_metrics"]
    client_mean_thrus = []
    client_var_thrus = []
    for client in client_metrics:
        client_mean_thrus.append(np.mean(client["thrus"]))
        client_var_thrus.append(np.var(client["thrus"]))
    mean_thru = np.sum(client_mean_thrus)
    std_thru = np.sqrt(np.sum(client_var_thrus))
    return (mean_thru, std_thru)


def extract_all_latencies(results_json):
    client_metrics = results_json["client_metrics"]
    latencies = []
    for client in client_metrics:
        for l in client["all_lats"]:
            cur_lats = json.loads(l)
            latencies.append(cur_lats)
    all_lats = np.array(latencies).flatten()
    return all_lats


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




def load_end_to_end_experiment(name, results_dir):
    """
    Parameters
    ----------
    name : str
        Experiment name (e.g. max_thru). Mostly used to label a line on the plot.
    results_dir : str
        Path to a directory containing all of the JSON results files for this experiment.
        Generally, each file represents an end-to-end run of the experiment with a fixed number
        of replicas per model. The different files in the directory hold everything else fixed
        and vary the number of model replicas.

    Returns
    --------
    (DataFrame, dict) :
        Each row in the dataframe corresponds to a single run (single file) of the
        experiment. The dict is a map from file name to loaded and filtered results dict loaded
        from the JSON.
    """
    experiments = load_end_to_end_results(results_dir)
    thru_means = []
    thru_stds = []
    costs = []
    latencies = []
    names = []
    configs = []
    p99s = []
    p95s = []

    for e in experiments:
        names.append(name)
        thru_mean, thru_std = get_throughput(experiments[e])
        thru_means.append(thru_mean)
        thru_stds.append(thru_std)
        costs.append(compute_cost(experiments[e]))
        all_lats = extract_all_latencies(experiments[e])
        latencies.append(all_lats)
        configs.append(experiments[e]["node_configs"])
        p99s.append(np.percentile(all_lats, 99))
        p95s.append(np.percentile(all_lats, 95))

    results_dict = {
        "mean_throughput": thru_means,
        "standard_dev_throughput": thru_stds,
        "latency": latencies,
        "p99_latency": p99s,
        "p95_latency": p95s,
        "cost": costs,
        "name": names,
        "config": configs,
    }

    df = pd.DataFrame.from_dict(results_dict)
    return (df, experiments)


# def load_all():
#
#     df1 = collect_results("max_thru",
#                           os.path.abspath("../results/e2e_profs/resnet_cascade/max_thru/"))
#     df2 = collect_results("min_lat",
#                           os.path.abspath("../results/e2e_profs/resnet_cascade/min_lat/"))
#
#     df3 = collect_results("slo_500",
#                           os.path.abspath("../results/e2e_profs/resnet_cascade/slo_500ms/"))
#
#     df = pd.concat([df1, df2, df3])
#     return df

























