import numpy as np
import json
import os
# import sys
import pandas as pd

from utils import COST_PER_GPU, COST_PER_CPU


def load_results(results_dir):
    fs = os.listdir(results_dir)
    experiments = {}
    for exp in fs:
        if exp[-4:] == "json":
            with open(os.path.join(results_dir, exp), "r") as f:
                data = json.load(f)
                # assert data["node_configs"][0]["no_diverge"]
                format_client_metrics(data)
                first_good_trial, last_good_trial = select_valid_trials(data)
                extract_good_results(data, first_good_trial, last_good_trial)
                if max([len(cm["thrus"]) for cm in data["client_metrics"]]) > 5:
                    experiments[exp] = data
        else:
            # print("skipping %s" % os.path.join(results_dir, exp))
            pass
    return experiments


# def skip_first_trial(results_json):
#     client_metrics = results_json["client_metrics"]
#     for client in client_metrics:
#         for metric in client:
#             client[metric] = client[metric][1:]


# heuristic to determine when latencies have flattened out
def select_valid_trials(results_json):
    p99_lats = results_json["client_metrics"][0]["p99_lats"]

    num_good_trials = 5

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
    return max(0, len(p99_lats) - num_good_trials - 1), len(p99_lats) - 1


def extract_good_results(results_json, first_good_trial, last_good_trial):
    node_configs = results_json["node_configs"]
    results_json["node_configs"] = [n for n in node_configs if "request_delay" not in n]
    client_metrics = results_json["client_metrics"]
    for client in client_metrics:
        # First deal with inconsistently formatted all_lats list:
        lat_entries_per_trial = len(client["all_lats"]) / len(client["p99_lats"])
        if lat_entries_per_trial > 1:
            # print("Found {} queries per trial".format(lat_entries_per_trial))
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

def format_client_metrics(data):
    if type(data["client_metrics"]) == dict:
        data["client_metrics"] = [data["client_metrics"]]
    if "input_length_words" in data["node_configs"][0]:
        num_words = data["node_configs"][0]["input_length_words"]
        data["node_configs"].pop(0)
        data["node_configs"][0]["input_length_words"] = num_words




def num_gpus(exp):
    cloud, gpu_type = get_gpu_type(exp)
    if gpu_type == "none":
        return 0
    else:
        return 1


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


def get_gpu_type(exp):
    if "instance_type" in exp["node_configs"][0]:
        if "p2" in exp["node_configs"][0]["instance_type"]:
            if exp["node_configs"][0]["gpus_per_replica"] > 0:
                return ("aws", "k80")
            else:
                return ("aws", "none")
        elif "p3" in exp["node_configs"][0]["instance_type"]:
            if exp["node_configs"][0]["gpus_per_replica"] > 0:
                return ("aws", "v100")
            else:
                return ("aws", "none")
        else:
            print("Error: unknown GPU type for instance type {}".format(
                exp["node_configs"][0]["instance_type"]))
            return None
    elif "cloud" in exp["node_configs"][0]:
        return (exp["node_configs"][0]["cloud"], exp["node_configs"][0]["gpu_type"])
    else:
        print("Error: unknown cloud for exp:\n{}".format(json.dumps(exp["node_configs"][0], indent=2)))


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
    # if "clipper_metrics" in exp:
    #     return extract_client_metrics_old(exp)
    # else:
    return extract_client_metrics_new(exp)


def compute_cost(results_json):
    # TODO: update cost to factor in varying GPU costs
    nodes = results_json["node_configs"]
    total_cost = 0.0
    for n in nodes:
        num_reps = n["num_replicas"]
        n_gpus = num_gpus(results_json) * num_reps
        n_cpus = num_cpus(results_json) * num_reps
        cost = float(n_gpus) * COST_PER_GPU + float(n_cpus) * COST_PER_CPU
        total_cost += cost
    return total_cost


def extract_all_latencies(results_json):
    client_metrics = results_json["client_metrics"]
    latencies = []
    for client in client_metrics:
        for l in client["all_lats"]:
            cur_lats = json.loads(l)
            latencies.append(cur_lats)
    all_lats = np.array(latencies).flatten()
    return all_lats


def create_model_profile_df(results_dir):
    experiments = load_results(results_dir)
    if len(experiments) == 0:
        return None

    gpus = []
    cpus = []
    mean_thrus = []
    std_thrus = []
    p99_lats = []
    mean_batches = []
    # inst_types = []
    costs = []
    fnames = []
    clouds = []
    gpu_types = []

    model_name = experiments[next(iter(experiments))]["node_configs"][0]["name"]

    for fname, e in experiments.items():
        # inst_types.append(instance_type(e))
        # gpus.append(num_gpus(e))
        cpus.append(num_cpus(e))
        mean_thru, std_thru = throughput(e)
        mean_thrus.append(mean_thru)
        std_thrus.append(std_thru)
        costs.append(compute_cost(e))
        all_lats = extract_all_latencies(e)
        model_p99_lat, mean_batch = extract_client_metrics(e)
        p99_lats.append(np.percentile(all_lats, 99))
        mean_batches.append(mean_batch)
        cloud, gpu_type = get_gpu_type(e)
        clouds.append(cloud)
        gpu_types.append(gpu_type)
        if gpu_type == "none":
            gpus.append(0)
        else:
            gpus.append(1)
        fnames.append(fname)

    results_dict = {
        # "num_gpus_per_replica": gpus,
        "num_cpus_per_replica": cpus,
        "mean_throughput_qps": mean_thrus,
        "std_throughput_qps": std_thrus,
        "p99_latency": p99_lats,
        "mean_batch_size": mean_batches,
        # "inst_type": inst_types,
        "cost": costs,
        "fname": fnames,
        "cloud": clouds,
        "gpu_type": gpu_types,
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
            res = create_model_profile_df(fname)
            if res is not None:
                model_name, df = res
                profs[model_name] = df
    return profs
