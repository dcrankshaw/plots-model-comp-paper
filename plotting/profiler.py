# import numpy as np
import os
# import sys
import pandas as pd

import end_to_end_profiles as e2e_profs
import single_model_profiles as sm_profs


class LogicalPipeline(object):

    def __init__(self, root_node, paths):
        """
        root_node : str
            The name of any node that all queries get sent to. Used to estimate relative work of
            each node in the pipline.
        paths : list
            A list of tuples of nodes. Each tuple should represent a unique path through the
            pipeline. Used to estimate the max end-to-end latency.
            Outside the prototype this would be generated automatically from a graph,
            but we list them explicitly here to keep things simple.

        """
        self.root = root_node
        self.paths = paths


def get_node_work(exp, root_node):
    """
    Parameters
    ----------
    exp : dict
        A dict containing the results of an end-to-end experiment. The node
        work should be the same under any pipeline configuration, so the experiment
        JSON can be from any run and only needs to be computed once per logical
        pipeline.
    root_node : str
        The name of a node in the DAG that all queries are sent to.
        This is a hack to get the total number of queries sent because
        we don't currently record that (though we should).
    """
    for client in exp["client_metrics"]:
        if "all_metrics" in client:
            clipper_metrics = client["all_metrics"]
            break
    counts = {}
    model_names = [n["name"] for n in exp["node_configs"]]
    if root_node not in model_names:
        print("Root node not in pipeline config")
    for m in model_names:
        counts[m] = 0.0

    for trial in clipper_metrics:
        counters = trial["counters"]
        count_key = "model:{name}:1:num_predictions"
        for c in counters:
            for m in model_names:
                if list(c.keys())[0] == count_key.format(name=m):
                    counts[m] += float(c[count_key.format(name=m)]["count"])

    total_count = counts[root_node]
    for m in counts:
        counts[m] = counts[m] / total_count

    return counts


def get_node_perf_from_profile(profile, batch_size, num_gpus, num_cpus):
    """Finds the relevant row in a model profile for the provided physical
    configuration (batch size and resource bundle).

    Parameters
    ----------
    profile : DataFrame
        The model profile as loaded by single_model_profiles.load_single_model_profiles()
    batch_size : int
        Batch size configured for node
    num_gpus : int
        Num gpus for node (either 0 or 1)
    num_cpus : int
        Num cpus for node (should always be 1)
    """

    result = profile[(profile.num_gpus_per_replica == num_gpus)
                     & (profile.num_cpus_per_replica == num_cpus)
                     & (profile.mean_batch_size < (batch_size + batch_size * 0.2))
                     & (profile.mean_batch_size > (batch_size - batch_size * 0.2))
                     ]
    if len(result) > 1:
        print("Ambiguous profile setting")
    elif len(result) < 1:
        print("No profile found for: {g} gpus, {c} cpus, {b} batch size".format(
            g=num_gpus, c=num_cpus, b=batch_size))
    else:
        return result


def predict_performance_for_pipeline_config(node_configs, node_profs, logical_pipeline, node_work):
    """
    Parameters
    ----------
    node_configs : list of dicts
        Each element in the list is a dict containing the physical configuration of the node.
    node_profs : dict
        Single model profs dict as loaded by load_single_model_profiles()
    logical_pipeline : LogicalPipeline
        The logical pipeline structure
    node_work : dict
        Relative work each node in the pipeline performs. Provided by get_node_work().

    Returns
    -------
    cost of the configuration
    throughput of the configuration
    latency of the pipeline
    """
    paths = logical_pipeline.paths

    node_perfs = {}
    bottleneck_thru = -1
    total_cost = 0.0

    for node in node_configs:
        prof = node_profs[node["name"]]
        num_reps = node["num_replicas"]
        batch_size = node["batch_size"]
        num_gpus = node["gpus_per_replica"]
        num_cpus = node["cpus_per_replica"]
        expected_perf_prof = get_node_perf_from_profile(prof, batch_size, num_gpus, num_cpus)
        work = node_work[node["name"]]
        adjusted_throughput = expected_perf_prof.mean_throughput_qps.tolist()[0] / work * num_reps
        p99_lat = expected_perf_prof.p99_latency_ms.tolist()[0]
        cost = expected_perf_prof.cost.tolist()[0] * num_reps
        total_cost += cost
        node_perfs[node["name"]] = {"cost": cost, "p99_lat": p99_lat, "thru": adjusted_throughput}

        if bottleneck_thru == -1:
            bottleneck_thru = adjusted_throughput
        bottleneck_thru = min(bottleneck_thru, adjusted_throughput)

    longest_path_latency = 0
    for path in paths:
        path_latency = 0
        for node in path:
            path_latency += node_perfs[node]["p99_lat"]
        longest_path_latency = max(longest_path_latency, path_latency)

    return pd.Series([bottleneck_thru, longest_path_latency / 1000.0],
                     index=["estimated_thru", "estimated_latency"])


def estimate_end_to_end_exp(name, pipeline, empirical_results_df, experiments, single_model_profs):
    node_work = get_node_work(experiments[next(iter(experiments))], pipeline.root)

    def apply_func(row):
        node_configs = row["config"]
        return predict_performance_for_pipeline_config(node_configs,
                                                       single_model_profs,
                                                       pipeline,
                                                       node_work)

    return empirical_results_df.apply(apply_func, axis=1)


def get_logical_pipeline(pipeline_name):
    # Image driver 1
    if pipeline_name == "pipeline_one":
        paths = [("tf-resnet-feats", "tf-kernel-svm"),
                 ("inception", "tf-log-reg")]
        root_node = "inception"
        return LogicalPipeline(root_node, paths)

    # Resnet Cascade
    elif pipeline_name == "pipeline_three":
        paths = [("alexnet",),
                 ("alexnet", "res50"),
                 ("alexnet", "res50", "res152")]
        root_node = "alexnet"
        return LogicalPipeline(root_node, paths)


def load_pipeline_three_systemx():
    pipeline = get_logical_pipeline("pipeline_three")
    single_model_profs = sm_profs.load_single_model_profiles()
    dirpath = os.path.abspath("../results/e2e_profs/systemx/resnet_cascade")
    exp_dfs = []
    for d in os.listdir(dirpath):
        df, raw_results = e2e_profs.load_end_to_end_experiment(d, os.path.join(dirpath, d))
        df = df.merge(estimate_end_to_end_exp(d, pipeline, df, raw_results, single_model_profs),
                      left_index=True, right_index=True)
        exp_dfs.append(df)

    exp_three_dfs = pd.concat(exp_dfs)
    exp_three_dfs = exp_three_dfs[["name",
                                   "mean_throughput",
                                   "estimated_thru",
                                   "p99_latency",
                                   "estimated_latency",
                                   "cost"]]
    return exp_three_dfs


def load_pipeline_one_systemx():
    pipeline = get_logical_pipeline("pipeline_one")
    single_model_profs = sm_profs.load_single_model_profiles()
    dirpath = os.path.abspath("../results/e2e_profs/systemx/image_driver_1")
    exp_dfs = []
    for d in os.listdir(dirpath):
        df, raw_results = e2e_profs.load_end_to_end_experiment(d, os.path.join(dirpath, d))
        df = df.merge(estimate_end_to_end_exp(d, pipeline, df, raw_results, single_model_profs),
                      left_index=True, right_index=True)
        exp_dfs.append(df)

    exp_three_dfs = pd.concat(exp_dfs)
    exp_three_dfs = exp_three_dfs[["name",
                                   "mean_throughput",
                                   "estimated_thru",
                                   "p99_latency",
                                   "estimated_latency",
                                   "cost"]]
    return exp_three_dfs
