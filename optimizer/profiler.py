# import numpy as np
import os
# import sys
import pandas as pd
import numpy as np

# import end_to_end_profiles as e2e_profs
import single_node_profiles as snp
import networkx as nx


class LogicalDAG(object):

    SOURCE = "SOURCE"
    SINK = "SINK"

    def __init__(self, adj_list, reference_node):
        """
        Parameters:
        -----------
        adj_list : dict
            The DAG as represented by an adjacency list. Every DAG is required
            to have two special nodes, SOURCE and SINK. SOURCE must have at least
            one outgoing edge and no incoming edges. SINK must have at least one incoming
            edge and no outgoing edges. There must be a path from the SOURCE to every node
            in the graph and a path from every node in the graph to the SINK.
        reference_node : str
            Can be any node in the graph that all queries are sent to. This is used
            to calculate the scale factor of the rest of the nodes in the graph.

        """
        assert len(adj_list[LogicalDAG.SOURCE]) > 0 and len(adj_list[LogicalDAG.SINK]) == 0
        self.adj_list = adj_list
        graph = nx.DiGraph()
        for parent in adj_list:
            for child in adj_list[parent]:
                graph.add_edge(parent, child)
        self.nx_graph = graph
        self.reference_node = reference_node

    def get_nx_graph(self):
        return self.nx_graph

    def enumerate_paths(self):
        """
        Returns:
        --------
        Every unique path through the DAG. 
        """

        paths = []
        
        def search(current_path, node):
            current_path.append(node)
            # Base case
            if node == LogicalDAG.SINK:
                paths.append(tuple(current_path))
            else:
                for next_node in self.adj_list[node]:
                    # We slice the list to copy it so each path gets its own copy
                    search(current_path[:], next_node)

        search([], LogicalDAG.SOURCE)
        assert(len(paths) >= 1)
        return paths

    def nodes(self):
        return self.adj_list.keys()


class NodeConfig(object):

    def __init__(self, name, num_cpus, gpu_type, batch_size, num_replicas, cloud):
        """
        num_cpus : int
            The number of virtual cpus allocated to this node
        gpu_type : str
            Which type of GPU this node is using. Can be None, "p100", "k80", "v100".
        batch_size : int
            The batch size for the node
        num_replicas : int
            The number of replicas of the node
        cloud : str
            The cloud service that was used. Can be either "gcp" or "aws".
        """
        self.name = name
        self.num_cpus = num_cpus
        self.gpu_type = gpu_type
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.cloud = cloud


class NodeProfile(object):
    
    def __init__(self, profile):
        self.profile = profile
    
    def estimate_performance(self, config):
        """
        Estimates the node's performance under the specified configuration.
        
        Parameters:
        -----------
        
        Returns:
        --------
        tuple : (p99_latency, throughput, cost)
            Returns estimated latency, throughput, and cost for this configuration.
            If there is not an exact batch size match, the profiler will perform linear
            interpolation.
            
        Raises:
        -------
        A RuntimeException will be raised if the node has not been profiled under the requested configuration.
        """
        resource_bundle_matches = self.profile[(self.profile.gpu_type == config.gpu_type)
                                             & (self.profile.num_cpus_per_replica == config.num_cpus)
                                             & (self.profile.cloud == config.cloud)]
        resource_bundle_matches = resource_bundle_matches.sort_values("mean_batch_size")
        glb = resource_bundle_matches['mean_batch_size'] <= config.batch_size
        lub = resource_bundle_matches['mean_batch_size'] >= config.batch_size
        idx_glb = resource_bundle_matches.loc[resource_bundle_matches.index[glb], 'mean_batch_size'].idxmax()
        idx_lub = resource_bundle_matches.loc[resource_bundle_matches.index[lub], 'mean_batch_size'].idxmin()
        relevant_entries = resource_bundle_matches.loc[idx_glb:idx_lub]
        assert np.all(np.diff(relevant_entries["mean_throughput_qps"]) > 0)
        estimated_thruput = np.interp(config.batch_size,
                                      relevant_entries["mean_batch_size"],
                                      relevant_entries["mean_throughput_qps"])
        estimated_thruput = estimated_thruput * config.num_replicas
        
        assert np.all(np.diff(relevant_entries["p99_latency"]) > 0)
        estimated_latency = np.interp(config.batch_size,
                                      relevant_entries["mean_batch_size"],
                                      relevant_entries["p99_latency"])
        # The cost for all the entries with the same resource bundle is the same,
        # so we just get it from the first entry
        cost = relevant_entries["cost"].iloc[0] * config.num_replicas
        return (estimated_latency, estimated_thruput, cost)
        

def get_logical_pipeline(pipeline_name):
    # paths = [("tf-resnet-feats", "tf-kernel-svm"),
    #          ("inception", "tf-log-reg")]
    # root_node = "inception"
    if pipeline_name == "pipeline_one":
        adj_list = {
            LogicalDAG.SOURCE : ["tf-resnet-feats", "inception"],
            "tf-resnet-feats": ["tf-kernel-svm",],
            "tf-kernel-svm": [LogicalDAG.SINK],
            "inception": ["tf-log-reg",],
            "tf-log-reg": [LogicalDAG.SINK],
            LogicalDAG.SINK: []
        }
        return LogicalDAG(adj_list, "inception")

    if pipeline_name == "pipeline_two":
        # paths = [("tf-lang-detect",),
        #          ("tf-lang-detect", "tf-lstm"),
        #          ("tf-lang-detect", "tf-nmt", "tf-lstm")]
        # root_node = "tf-lang-detect"
        adj_list = {
            LogicalDAG.SOURCE : ["tf-lang-detect",],
            "tf-lang-detect": ["tf-lstm", "tf-nmt", LogicalDAG.SINK],
            "tf-nmt": ["tf-lstm",],
            "tf-lstm": [LogicalDAG.SINK,],
            LogicalDAG.SINK: []
        }
        return LogicalDAG(adj_list, "tf-lang-detect")

    # Resnet Cascade
    elif pipeline_name == "pipeline_three":
        adj_list = {
            LogicalDAG.SOURCE : ["alexnet",],
            "alexnet": ["res50", LogicalDAG.SINK],
            "res50": ["res152", LogicalDAG.SINK],
            "res152": [LogicalDAG.SINK],
            LogicalDAG.SINK: []
        }
        return LogicalDAG(adj_list, "alexnet")


def get_node_scale_factors(exp, reference_node):
    """
    Parameters
    ----------
    exp : dict
        A dict containing the results of an end-to-end experiment. The node
        scale_factor should be the same under any pipeline configuration, so the experiment
        JSON can be from any run and only needs to be computed once per logical
        pipeline.
    reference_node : str
        The name of a node in the DAG that all queries are sent to.
        This is a workaround to get the total number of queries sent because
        we don't currently record that (though we should).
    """
    for client in exp["client_metrics"]:
        if "all_metrics" in client:
            clipper_metrics = client["all_metrics"]
            break
    counts = {}
    node_names = [n["name"] for n in exp["node_configs"]]
    if reference_node not in node_names:
        print("reference node not in pipeline config")
    for m in node_names:
        counts[m] = 0.0

    for trial in clipper_metrics:
        counters = trial["counters"]
        count_key = "model:{name}:1:num_predictions"
        for c in counters:
            for m in node_names:
                if list(c.keys())[0] == count_key.format(name=m):
                    counts[m] += float(c[count_key.format(name=m)]["count"])

    total_count = counts[reference_node]
    for m in counts:
        counts[m] = counts[m] / total_count

    return counts

def estimate_pipeline_performance_for_config(dag,
                         scale_factors,
                         node_configs,
                         single_node_profiles):
    """
    Estimate the end to end performance for a pipeline under a
    specific configuration.
    
    dag : LogicalDAG
        The logical pipeline structure
    scale_factors : dict
        A dict with the scale factors for each node in the pipeline
    node_configs : dict(str, NodeConfig)
        A dict with the physical configurations for each node in the pipeline.
    single_node_profiles : dict (str, NodeProfile)
        A dict with the profiles for each node in the pipeline.
    
    Returns:
    --------
    tuple : (p99_latency, throughput, cost)
        Returns estimated latency, throughput, and cost for the pipeline
        under the specified configuration (and workload via the scale factors).
    """
    paths = dag.enumerate_paths()
    bottleneck_thruput = None
    total_cost = 0.0
    max_latency = None
    for path in paths:
        path_latency = 0
        for node in path:
            # The source and sink nodes don't contribute to perf so
            # we skip them
            if node == LogicalDAG.SOURCE or node == LogicalDAG.SINK:
                continue
            prof = single_node_profiles[node]
            conf = node_configs[node]
            lat, thru, cost = prof.estimate_performance(conf)
            scaled_thru = thru / scale_factors[node]
            path_latency += lat
            if bottleneck_thruput is None:
                bottleneck_thruput = scaled_thru
            bottleneck_thruput = min(bottleneck_thruput, scaled_thru)
            total_cost += cost
        # Update latency at the end of the path
        if max_latency is None:
            max_latency = path_latency
        max_latency = max(max_latency, path_latency)
    return {
                "latency": max_latency,
                "throughput": bottleneck_thruput,
                "cost": total_cost
            }



def get_node_configs_from_experiment(exp):
    """
    Extract the physical node configs from an end-to-end pipeline run.

    Parameters
    ----------
    exp : dict
        A dict containing the results of an end-to-end experiment.       

    Returns
    -------
    dict(str, NodeConfig)
        A dict of NodeConfig objects that can be used by the optimizer to
        estimate the end to end performance for this configuration of the pipeline.
    """

    raw_configs = exp["node_configs"]
    node_configs = {}

    for node in raw_configs:
        name = node["name"]
        num_replicas = node["num_replicas"]
        num_cpus = node["cpus_per_replica"]
        batch_size = node["batch_size"]
        cloud, gpu_type = snp.get_gpu_type(node)
        node_configs[name] = NodeConfig(name,
                num_cpus,
                gpu_type,
                batch_size,
                num_replicas,
                cloud)

    return node_configs
        




# def find_closest_batch_size_index(batch_size, profile):
#     closest_batch = -1
#
#     for index, row in profile.iterrows():
#         if closest_batch == -1:
#             closest_batch = row["mean_batch_size"]
#         else:
#             if abs(batch_size - closest_batch) > abs(batch_size - row["mean_batch_size"]):
#                 closest_batch = row["mean_batch_size"]
#     return closest_batch
#
#
# def get_node_perf_from_profile(name, profile, batch_size, gpu_type, num_cpus):
#     """Finds the relevant row in a node profile for the provided physical
#     configuration (batch size and resource bundle).
#
#     Parameters
#     ----------
#     profile : DataFrame
#         The node profile as loaded by single_node_profiles.load_single_node_profiles()
#     batch_size : int
#         Batch size configured for node
#     gpu_type : String
#         Which gpu was used (can also be None)
#     num_cpus : int
#         Num cpus for node (should always be 1)
#     """
#
#     result = profile[(profile.gpu_type == gpu_type)
#                      & (profile.num_cpus_per_replica == num_cpus)
#                      & (profile.mean_batch_size <= (batch_size))
#                      & (profile.mean_batch_size > (batch_size - 0.1))
#                      ]
#     if len(result) < 1:
#         closest_batch = find_closest_batch_size_index(batch_size, profile[
#             (profile.gpu_type == gpu_type) &
#             (profile.num_cpus_per_replica == num_cpus)])
#         result = profile[(profile.gpu_type == gpu_type)
#                          & (profile.num_cpus_per_replica == num_cpus)
#                          & (profile.mean_batch_size <= (closest_batch + 0.1))
#                          & (profile.mean_batch_size > (closest_batch - 0.1))
#                          ]
#
#         # print(("No profile found for {m}: {g} gpus, {c} cpus, batch size {b}."
#         #       " Approximating with batch size {ab}").format(
#         #     m=name, g=num_gpus, c=num_cpus, b=batch_size, ab=closest_batch))
#     if len(result) > 1:
#         print("Ambiguous profile setting for batch size {b}".format(b=batch_size))
#         print(result)
#     if len(result) < 1:
#         print("Something weird happened")
#         print(profile)
#     else:
#         return result
#
#
# def predict_performance_for_pipeline_config(node_configs, node_profs, logical_pipeline, node_scale_factors):
#     """
#     Parameters
#     ----------
#     node_configs : list of dicts
#         Each element in the list is a dict containing the physical configuration of the node.
#     node_profs : dict
#         Single node profs dict as loaded by load_single_node_profiles()
#     logical_pipeline : LogicalPipeline
#         The logical pipeline structure
#     node_scale_factors : dict
#         Relative scale_factor each node in the pipeline performs. Provided by get_node_scale_factors().
#
#     Returns
#     -------
#     cost of the configuration
#     throughput of the configuration
#     latency of the pipeline
#     """
#     paths = logical_pipeline.paths
#
#     node_perfs = {}
#     bottleneck_thru = -1
#     total_cost = 0.0
#
#     for node in node_configs:
#         prof = node_profs[node["name"]]
#         num_reps = node["num_replicas"]
#         batch_size = node["batch_size"]
#         gpu_type = node["gpu_type"]
#         num_cpus = node["cpus_per_replica"]
#         expected_perf_prof = get_node_perf_from_profile(node["name"], prof, batch_size,
#                                                         gpu_type, num_cpus)
#         scale_factor = node_scale_factors[node["name"]]
#         adjusted_throughput = expected_perf_prof.mean_throughput_qps.tolist()[0] / scale_factor * num_reps
#         p99_lat = expected_perf_prof.p99_latency.tolist()[0]
#         cost = expected_perf_prof.cost.tolist()[0] * num_reps
#         total_cost += cost
#         node_perfs[node["name"]] = {"cost": cost, "p99_lat": p99_lat, "thru": adjusted_throughput}
#
#         if bottleneck_thru == -1:
#             bottleneck_thru = adjusted_throughput
#         bottleneck_thru = min(bottleneck_thru, adjusted_throughput)
#
#     longest_path_latency = 0
#     for path in paths:
#         path_latency = 0
#         for node in path:
#             path_latency += node_perfs[node]["p99_lat"]
#         longest_path_latency = max(longest_path_latency, path_latency)
#
#     return pd.Series([bottleneck_thru, longest_path_latency],
#                      index=["estimated_thru", "estimated_latency"])
#
#
# def estimate_end_to_end_exp(name, pipeline, empirical_results_df, experiments, single_node_profs):
#     node_scale_factors = get_node_scale_factors(experiments[next(iter(experiments))], pipeline.root)
#
#     def apply_func(row):
#         node_configs = row["config"]
#         return predict_performance_for_pipeline_config(node_configs,
#                                                        single_node_profs,
#                                                        pipeline,
#                                                        node_scale_factors)
#
#     return empirical_results_df.apply(apply_func, axis=1)
#
#
# def load_pipeline_systemx(pipeline, dirpath):
#     single_node_profs = sm_profs.load_single_node_profiles()
#     exp_dfs_list = []
#     for d in os.listdir(dirpath):
#         if d == "max_thru_hand_tuned":
#             print("Skipping {}".format(d))
#             continue
#         # if d == "alexnet_cpu_max_thru":
#         #     print("Skipping {}".format(d))
#         #     continue
#         df, raw_results = e2e_profs.load_end_to_end_experiment(d, os.path.join(dirpath, d))
#         df = df.merge(estimate_end_to_end_exp(d, pipeline, df, raw_results, single_node_profs),
#                       left_index=True, right_index=True)
#         exp_dfs_list.append(df)
#
#     exp_dfs = pd.concat(exp_dfs_list)
#     exp_dfs = exp_dfs[["name",
#                        "mean_throughput",
#                        "estimated_thru",
#                        "p99_latency",
#                        "p95_latency",
#                        "estimated_latency",
#                        "cost", "latency"]]
#     return exp_dfs
#
#
# def load_pipeline_three_systemx():
#     pipeline = get_logical_pipeline("pipeline_three")
#     dirpath = os.path.abspath("../results/e2e_profs/systemx/resnet_cascade")
#     return load_pipeline_systemx(pipeline, dirpath)
#
#
# def load_pipeline_one_systemx():
#     pipeline = get_logical_pipeline("pipeline_one")
#     dirpath = os.path.abspath("../results/e2e_profs/systemx/image_driver_1")
#     return load_pipeline_systemx(pipeline, dirpath)
