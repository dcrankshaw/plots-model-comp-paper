import sys
import os
import json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath("../optimizer/"))
import utils


##########################################################
################## GENERAL UTILITIES #####################

def get_lam_and_cv_from_fname(arrival_process_fname):

    lam_and_cv = arrival_process_fname.rstrip(".deltas")
    if "_" in lam_and_cv:
        lam = int(lam_and_cv.split("_")[0])
        cv = float(lam_and_cv.split("_")[1])
    else:
        cv = 1.0
        lam = int(lam_and_cv)
    return (lam, cv)





##########################################################
################# TENSORFLOW SERVING #####################

def compute_tfs_cost(results):
    node_configs = results["node_configs"]
    cost = 0
    for n in node_configs:
        # Divide by 2 to convert from virtual cpus to physical cpus
        num_cpus = n["cpus_per_replica"] / 2 * n["num_replicas"]
        num_gpus = n["gpus_per_replica"] * n["num_replicas"]
        if num_gpus > 0:
            if "p3" in n["instance_type"]:
                gpu_type = "v100"
            elif "p2" in n["instance_type"]:
                gpu_type = "k80"
        else:
            gpu_type = "none"
        cost += utils.get_cpu_cost("aws", num_cpus) + utils.get_gpu_cost("aws", gpu_type, num_gpus)
    return cost


def compute_tfs_slo_miss_rate(results, slo):
    # Skip the first trial
    lats = np.array(results["client_metrics"][0]["all_lats"][1:]).flatten()
    slo_miss_rate = np.sum(lats > slo) / len(lats)
    return slo_miss_rate


def load_tfs_results(path, slo):
    with open(path, "r") as f:
        results = json.load(f)

    arrival_process_fname = os.path.basename(results["arrival_process"]["filename"])
    lam, cv = get_lam_and_cv_from_fname(arrival_process_fname)
    cost = compute_tfs_cost(results)
    slo_miss_rate = compute_tfs_slo_miss_rate(results, slo)
    slo_plus_25_miss_rate = compute_tfs_slo_miss_rate(results, slo*1.25)
    return {"cost": cost, "lambda": lam, "CV": cv, "slo_miss_rate": slo_miss_rate,
            "slo_plus_25_per_miss_rate": slo_plus_25_miss_rate}

##########################################################
################ SINGLE PROCESS DRIVER ###################

def compute_spd_cost(results):
    node_configs = results["node_configs"]
    num_models = len(node_configs)
    # All nodes use the same cpu set, so only get cpus once outside the for loop
    # Divide by 2 to convert from virtual cpus to physical cpus
    num_reps = node_configs[0]["num_replicas"]
    # # cpus_per_rep = len(node_configs[0]["allocated_cpus"][0].split(" ")) / 2
    # # print(cpus_per_rep)
    # assert cpus_per_rep == 4
    # num_cpus = num_reps * cpus_per_rep
    gpu_type = "v100"
    # # Check how many gpus were used
    # num_gpus = 0
    # for n in node_configs:
    #     if len(n["gpus"]) > 0:
    #         gpus_per_rep = len(n["gpus"][0].split(" "))
    #         num_gpus = gpus_per_rep * num_reps
    #         break
    num_cpus = 4 * num_reps
    num_gpus = 2 * num_reps
    cost = utils.get_cpu_cost("aws", num_cpus) + utils.get_gpu_cost("aws", gpu_type, num_gpus)
    return cost

def compute_spd_slo_miss_rate(results, slo):
    # Skip the first trial
    lats = np.array(results["client_metrics"][0]["all_lats"][1:])
    lats = np.hstack(lats)
    slo_miss_rate = np.sum(lats > slo) / len(lats)
    # print(np.sum(lats > 100000))
    return slo_miss_rate

def compute_spd_thruput(results, lam):
    thrus = results["client_metrics"][0]["thrus"][1:]
    thru = np.mean(thrus)
    return thru, lam-thru

def load_spd_run(path, slo, provision_strategy):
    """
    Loads a single SPD experiment JSON file
    """
    with open(path, "r") as f:
        results = json.load(f)
    arrival_process_fname = os.path.basename(results["arrival_process"]["file_path"])
    # lam, cv = get_lam_and_cv_from_fname(arrival_process_fname)
    lam = results["experiment_config"]["lambda_val"]
    cv = results["experiment_config"]["cv"]
    cost = compute_spd_cost(results)
    slo_miss_rate = compute_spd_slo_miss_rate(results, slo)
    thruput, thruput_delta = compute_spd_thruput(results, lam)
    slo_plus_25_miss_rate = compute_spd_slo_miss_rate(results, slo*1.25)
    return {
            "name": "SPD-{}".format(provision_strategy),
            "cost": cost,
            "lambda": lam,
            "CV": cv,
            "slo": slo,
            "slo_miss_rate": slo_miss_rate,
            "slo_plus_25_per_miss_rate": slo_plus_25_miss_rate,
            "throughput": thruput,
            "lam_minus_through": thruput_delta
            }

def load_spd_experiment(path, slo, provision_strategy):
    results = []
    expected_num_results = len(list(os.listdir(path)))
    for reps_dir in os.listdir(path):
        for f in os.listdir(os.path.join(path, reps_dir)):
            if "results" in f and f[-4:] == "json":
                results.append(
                        load_spd_run(
                            os.path.join(path, reps_dir, f),
                            slo,
                            provision_strategy))
    return results


def get_slo_and_cv_from_spd_dirname(f):
    # File name structure is EITHER:
    # 1) <slo-in-ms-as-int>ms_cv<cv-as-float>
    # 2) min_lat_cv<cv-as-float>
    if "min_lat" in f:
        slo = "min_lat"
    else:
        # Convert to seconds
        slo = float(f.split("ms")[0]) / 1000.0
    cv = float(f.split("cv")[1])
    return slo, cv


def load_spd_pipeline_one():
    base_path = os.path.abspath("../SPD/image_driver_1/v100-8xlarge")
    all_results = []

    for d in os.listdir(base_path):
        if "INCOMPLETE" in d:
            continue
        slo, cv = get_slo_and_cv_from_spd_dirname(d)
        if slo == "min_lat":
            for slo in [.35, .5, 1.0]:
                all_results.extend(load_spd_experiment(
                    os.path.join(base_path, d, "mean_provision"),
                    slo,
                    "min_lat_mean_provision"))
                all_results.extend(load_spd_experiment(
                    os.path.join(base_path, d, "peak_provision"),
                    slo,
                    "min_lat_peak_provision"))
        else:
            all_results.extend(load_spd_experiment(
                os.path.join(base_path, d, "mean_provision"),
                slo,
                "mean_provision"))
            all_results.extend(load_spd_experiment(
                os.path.join(base_path, d, "peak_provision"),
                slo,
                "peak_provision"))
    return pd.DataFrame(all_results)
        


##########################################################
####################### INFERLINE ########################

def compute_inferline_cost(results, conf_name):
    # node_configs = results["loaded_config"]["node_configs"]
    node_configs = results[conf_name]["node_configs"]
    cost = 0
    for name, n in node_configs.items():
        num_replicas = n["num_replicas"]
        num_cpus = n["num_cpus"] * num_replicas
        gpu_type = n["gpu_type"]
        cloud = n["cloud"]
        cost += utils.get_cpu_cost(cloud, num_cpus) + utils.get_gpu_cost(cloud, gpu_type, num_replicas)
    return cost

def compute_throughput_inferline(results, lam):
    thrus = []
    trials = results["throughput_results"]["summary_metrics"][1:]
    for t in trials:
        thrus.append(t["client_thrus"]["e2e"])
    thru = np.mean(thrus)
    return thru, lam - thru

def load_inferline_results_file(path):
    if path[-4:] != "json":
        return None

    with open(path, "r") as f:
        results = json.load(f)
    conf_name = "used_config"
    cv = results[conf_name]["cv"]
    slo = results[conf_name]["slo"]
    lam = results[conf_name]["lam"]
    utilization = results[conf_name]["utilization"]
    trials = results["throughput_results"]["client_metrics"][0][1:]
    lats = []
    for t in trials:
        datalists = t["data_lists"]
        for d in datalists:
            if list(d.keys())[0] == "e2e:prediction_latencies":
                items = d["e2e:prediction_latencies"]["items"]
                for i in items:
                    lats.append(float(list(i.values())[0]) / 1000.0 / 1000.0)
    lats = np.array(lats)
    slo_miss_rate = np.sum(lats > slo) / len(lats)
    slo_plus_25_miss_rate = np.sum(lats > slo*1.25) / len(lats)
    thruput, thruput_delta = compute_throughput_inferline(results, lam)
    orig_cost = compute_inferline_cost(results, "orig_config")
    used_cost = compute_inferline_cost(results, "used_config")
    return {
            "name": "InferLine",
            "orig_cost": orig_cost,
            "used_cost": used_cost,
            "lambda": lam,
            "CV": cv,
            "slo": slo,
            "slo_miss_rate": slo_miss_rate,
            "slo_plus_25_per_miss_rate": slo_plus_25_miss_rate,
            "utilization": utilization,
            "throughput": thruput,
            "lam_minus_through": thruput_delta
            }

def load_all_inferline_sys_comp_results(base_path):
    all_results = []
    for exp in os.listdir(base_path):
        for fname in os.listdir(os.path.join(base_path, exp)):
            result = load_inferline_results_file(os.path.join(base_path, exp, fname))
            if result is not None:
                all_results.append(result)
    return all_results

def load_inferline_pipeline_one():
    # base_path = os.path.abspath("../results_cpp_benchmarker/e2e_results/image_driver_1/sys_comp/util_0.7")
    base_path = os.path.abspath("../results_cpp_benchmarker/e2e_results/image_driver_1/sys_comp_dynamic_replication/debug_util_1.0")
    loaded_exps = load_all_inferline_sys_comp_results(base_path)
    df = pd.DataFrame(loaded_exps)
    return df

##########################################################

def load_e2e_experiments(util, base_path):
    # experiments = [
    #     {"name": "Inferline", "slo": .35, "path": os.path.abspath("../results_cpp_benchmarker/e2e_results/image_driver_1/image_driver_one_slo_0.35_cv_1")},
    #     {"name": "TFS Peak", "slo": .35, "path": os.path.abspath("../TFS/image_driver_1/min_latency_arrival_process/v100-8xlarge/peak_350")},
    #     {"name": "TFS Mean", "slo": .35, "path": os.path.abspath("../TFS/image_driver_1/min_latency_arrival_process/v100-8xlarge/mean")},
    #     {"name": "SPD Mean Batching", "slo": .35, "path": os.path.abspath("../SPD/image_driver_1/v100-8xlarge/mean_provision_350/")},
    #     {"name": "SPD Peak Batching", "slo": .35, "path": os.path.abspath("../SPD/image_driver_1/v100-8xlarge/peak_provision_350/")},
    #     {"name": "SPD Mean No Batching", "slo": .35, "path": os.path.abspath("../SPD/image_driver_1/v100-8xlarge/mean_provision_min_lat/")},
    #     {"name": "SPD Peak No Batching", "slo": .35, "path": os.path.abspath("../SPD/image_driver_1/v100-8xlarge/peak_provision_min_lat/")},
    #     {"name": "Inferline", "slo": .50, "path": os.path.abspath("../results_cpp_benchmarker/e2e_results/image_driver_1/image_driver_one_slo_0.5_cv_1")},
    #     {"name": "TFS Mean", "slo": .50, "path": os.path.abspath("../TFS/image_driver_1/min_latency_arrival_process/v100-8xlarge/mean")},
    #     {"name": "TFS Peak", "slo": .50, "path": os.path.abspath("../TFS/image_driver_1/min_latency_arrival_process/v100-8xlarge/peak_500")},
    #     {"name": "SPD Mean No Batching", "slo": .50, "path": os.path.abspath("../SPD/image_driver_1/v100-8xlarge/mean_provision_min_lat/")},
    #     {"name": "SPD Peak No Batching", "slo": .50, "path": os.path.abspath("../SPD/image_driver_1/v100-8xlarge/peak_provision_min_lat/")},
    #     {"name": "Inferline", "slo": 1.0, "path": os.path.abspath("../results_cpp_benchmarker/e2e_results/image_driver_1/image_driver_one_slo_1.0_cv_1")}
    # ]
    #
    # loaded_exps = []
    #
    # for exp in experiments:
    #     for f in os.listdir(exp["path"]):
    #         if f[-4:] == "json":
    #             try:
    #                 loaded = exp.copy()
    #                 if "TFS" in exp["name"]:
    #                     loaded.update(load_tfs_results(os.path.join(exp["path"], f), exp["slo"]))
    #                 elif "SPD" in exp["name"]:
    #                     loaded.update(load_spd_results(os.path.join(exp["path"], f), exp["slo"]))
    #                 elif "Inferline" in exp["name"]:
    #                     loaded.update(load_inferline_results(os.path.join(exp["path"], f), exp["slo"]))
    #                 loaded_exps.append(loaded)
    #             except json.JSONDecodeError:
    #                 print("Could not load {}".format(os.path.join(exp["path"], f)))
    loaded_exps = load_all_inferline_sys_comp_results(util, base_path)
    df = pd.DataFrame(loaded_exps)
    return df

if __name__ == "__main__":
    print(load_e2e_experiments())
