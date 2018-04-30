import numpy as np
import json
import os
# import sys
import pandas as pd
from utils import get_cpu_cost, get_gpu_cost
import logging
from multiprocessing import Process, Queue
import traceback


logger = logging.getLogger(__name__)

def load_file(results_dir, exp, queue):
    result = None
    try:
        if exp[-4:] == "json":
            with open(os.path.join(results_dir, exp), "r") as f:
                try:
                    data = json.load(f)
                    for stage in ["latency_results", "throughput_results"]:
                        # if "gcp" in exp:
                        if type(data[stage]) == list and len(data[stage]) == 1:
                            data[stage] = data[stage][0]
                        # Remove first trial
                        data[stage]["summary_metrics"] = data[stage]["summary_metrics"][1:]
                        data[stage]["clipper_metrics"] = data[stage]["clipper_metrics"][2:]
                        data[stage]["client_metrics"] = data[stage]["client_metrics"][1:]
                        if "remote" in exp:
                            data["node_configs"][0]["instance_type"] = "p2.8xlarge"
                    result = (exp, data)
                except json.JSONDecodeError as e:
                    logger.error("Error loading {}: {}".format(f.name, e.msg))
    except Exception as e:
        traceback.print_exc(e)
    queue.put(result)
    return

def load_results(results_dir):
    fs = os.listdir(results_dir)
    experiments = {}
    queue = Queue()
    procs = []

    for exp in fs:
        p = Process(target=load_file, args=(results_dir, exp, queue))
        p.start()
        procs.append(p)
    for p in procs:
        result = queue.get()
        if result:
            exp, data = result
            experiments[exp] = data
        # p.join()
    return experiments


def num_gpus(exp):
    node_config = exp["node_configs"][0]
    cloud, gpu_type = get_gpu_type(node_config)
    if gpu_type == "none":
        return 0
    else:
        return 1


def num_cpus(exp):
    return exp["node_configs"][0]["cpus_per_replica"]


def configured_batch_size(exp):
    return exp["node_configs"][0]["batch_size"]


def node_name(exp):
    return exp["node_configs"][0]["name"]


def instance_type(exp):
    if "instance_type" in exp["node_configs"][0]:
        return exp["node_configs"][0]["instance_type"]
    else:
        return "g3.4xlarge"


def get_gpu_type(node_config):
    if "instance_type" in node_config:
        if "p2" in node_config["instance_type"]:
            if node_config["gpus_per_replica"] > 0:
                return ("aws", "k80")
            else:
                return ("aws", "none")
        elif "p3" in node_config["instance_type"]:
            if node_config["gpus_per_replica"] > 0:
                return ("aws", "v100")
            else:
                return ("aws", "none")
        elif "m4" in node_config["instance_type"]:
            # print("Client was running on separate machine. Assuming you used V100s")
            if node_config["gpus_per_replica"] > 0:
                return ("aws", "v100")
            else:
                return ("aws", "none")

        else:
            print("Error: unknown GPU type for instance type {}".format(
                node_config["instance_type"]))
            return None
    elif "cloud" in node_config:
        gpu_type = node_config["gpu_type"]
        if gpu_type is None:
            gpu_type = "none"
        return (node_config["cloud"], gpu_type)
    else:
        print("Error: unknown cloud for exp:\n{}".format(json.dumps(node_config, indent=2)))


def get_mean_throughput(exp, stage):
    name = node_name(exp)
    thrus = []
    for cm in exp[stage]["summary_metrics"]:
        thrus.append(float(cm["client_thrus"][name]))
    return (round(np.mean(thrus), 1), np.std(thrus))


def get_mean_batch_size(exp):
    name = node_name(exp)
    batch_sizes = []
    for cm in exp["latency_results"]["summary_metrics"]:
        batch_sizes.append(float(cm["batch_sizes"][name]))
    return (np.mean(batch_sizes), np.std(batch_sizes))


def get_mean_queue_size(exp):
    name = node_name(exp)
    queue_sizes = []
    for cm in exp["latency_results"]["summary_metrics"]:
        queue_sizes.append(float(cm["queue_sizes"][name]))
    return (np.mean(queue_sizes), np.std(queue_sizes))


def compute_cost(results_json):
    nodes = results_json["node_configs"]
    total_cost = 0.0
    for n in nodes:
        num_reps = n["num_replicas"]
        n_cpus = num_cpus(results_json) * num_reps
        cloud, gpu_type = get_gpu_type(n)
        cost = get_cpu_cost(cloud, n_cpus) + get_gpu_cost(cloud, gpu_type)
        # cost = float(n_gpus) * COST_PER_GPU + float(n_cpus) * COST_PER_CPU
        total_cost += cost
    return total_cost


def extract_all_latencies(results_json):
    client_metrics = results_json["latency_results"]["client_metrics"]
    name = node_name(results_json)
    latencies = []
    key_name = "{}:prediction_latencies".format(name)
    for trial in client_metrics:
        for l in trial["data_lists"]:
            if list(l.keys())[0] == key_name:
                cur_lats = [float(list(i.values())[0]) for i in l[key_name]["items"]]
                latencies.extend(cur_lats)
    all_lats = np.array(latencies).flatten() / 1000.0 / 1000.0  # Convert micros to secs
    return all_lats


def create_node_profile_df(results_dir):
    experiments = load_results(results_dir)
    if len(experiments) == 0:
        return None

    gpus = []
    cpus = []
    thru_stage_mean_thrus = []
    thru_stage_std_thrus = []
    latency_stage_mean_thrus = []
    latency_stage_std_thrus = []
    p99_lats = []
    mean_batches = []
    mean_queues = []
    # inst_types = []
    costs = []
    fnames = []
    clouds = []
    gpu_types = []
    contentions = []

    node_name = experiments[next(iter(experiments))]["node_configs"][0]["name"]

    for fname, e in experiments.items():
        cpus.append(num_cpus(e))
        thru_stage_mean_thru, thru_stage_std_thru = get_mean_throughput(e, "throughput_results")
        thru_stage_mean_thrus.append(thru_stage_mean_thru)
        thru_stage_std_thrus.append(thru_stage_std_thru)
        latency_stage_mean_thru, latency_stage_std_thru = get_mean_throughput(e, "latency_results")
        latency_stage_mean_thrus.append(latency_stage_mean_thru)
        latency_stage_std_thrus.append(latency_stage_std_thru)
        costs.append(compute_cost(e))
        all_lats = extract_all_latencies(e)
        p99_lats.append(np.percentile(all_lats, 99))
        mean_batch, std_batch = get_mean_batch_size(e)
        mean_batches.append(mean_batch)
        mean_queue, std_queue = get_mean_queue_size(e)
        mean_queues.append(mean_queue)
        node_config = e["node_configs"][0]
        cloud, gpu_type = get_gpu_type(node_config)
        clouds.append(cloud)
        gpu_types.append(gpu_type)
        if "contention" in e:
            contentions.append(e["contention"]["contention_throughput_qps"])
        else:
            contentions.append(-1)
        if gpu_type == "none":
            gpus.append(0)
        else:
            gpus.append(1)
        fnames.append(fname)

    results_dict = {
        # "num_gpus_per_replica": gpus,
        "num_cpus_per_replica": cpus,
        "thru_stage_mean_throughput_qps": thru_stage_mean_thrus,
        "thru_stage_std_throughput_qps": thru_stage_std_thrus,
        "latency_stage_mean_throughput_qps": latency_stage_mean_thrus,
        "latency_stage_std_throughput_qps": latency_stage_std_thrus,
        "p99_latency": p99_lats,
        "mean_batch_size": mean_batches,
        "mean_queue_size": mean_queues,
        # "inst_type": inst_types,
        "cost": costs,
        "fname": fnames,
        "cloud": clouds,
        "gpu_type": gpu_types,
        "contention": contentions,
    }

    df = pd.DataFrame.from_dict(results_dict)
    df = df[list(results_dict.keys())]
    return (node_name, df)


def load_single_node_profiles(
        single_node_profs_dir=os.path.abspath("../results_cpp_benchmarker/single_model_profs_contention_tuned/"),
        models=[]):
    """
    Returns
    -------
    dict :
        A dict that maps node name to a DataFrame containing single node profile for that
        node. The node name is set based on the node config in the experimental results. It will
        not necessarily match the directory name of the directory containing the profiling results
        for that node.
    """
    logger.info("Loading profs from {}".format(os.path.basename(single_node_profs_dir)))
    profs = {}
    for m in os.listdir(single_node_profs_dir):
        fname = os.path.join(single_node_profs_dir, m)
        if os.path.isdir(fname):
            if not any(m in fname for m in models):
                continue
            res = create_node_profile_df(fname)
            if res is not None:
                node_name, df = res
                profs[node_name] = df
    return profs
