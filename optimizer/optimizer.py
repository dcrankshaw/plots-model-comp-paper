import profiler
import itertools


class BruteForceOptimizer(object):
    def __init__(self, dag, scale_factors, node_profs):
        """
        Parameters
        ----------
        dag : profiler.LogicalDAG
            The logical pipeline structure
        scale_factors : dict
            A dict where the keys are the node names and the values are the scale factors.
            Produced by profiler.get_node_scale_factors().
        node_profs : dict
            A dict where the keys are the node names and the values are profiler.NodeProfile
            objects.
        """
        self.dag = dag
        self.scale_factors = scale_factors
        self.node_profs = node_profs

    def select_optimal_config(self,
                              cloud,
                              latency_constraint,
                              cost_constraint,
                              max_replication_factor=1):
        """
        cloud : str
            Can be "aws" or "gcp".
        latency_constraint : float
            The maximum p99 latency in seconds.
        cost_constraints : float
            The maximum pipeline cost in $/hr.
        max_replication_factor : int, optional
            The maximum number of replicas of any node that will be considered.
            Note that since this is a brute-force optimizer, increasing this will
            exponentially increase the search space and therefore search time.
        """
        all_node_configs = [self.node_profs[node].enumerate_configs(
            max_replication_factor=max_replication_factor) for node in self.dag.nodes()]
        all_pipeline_configs = itertools.product(*all_node_configs)
        best_config = None
        best_config_perf = None
        cur_index = 0
        for p_config in all_pipeline_configs:
            cur_index += 1
            if cur_index % 1000 == 0:
                print("Processed {}".format(cur_index))
            cur_node_configs = {n.name: n for n in p_config}
            if not profiler.is_valid_pipeline_config(cur_node_configs):
                continue
            if not list(cur_node_configs.values())[0].cloud == cloud:
                continue
            cur_config_perf = profiler.estimate_pipeline_performance_for_config(
                self.dag, self.scale_factors, cur_node_configs, self.node_profs)
            if (cur_config_perf["latency"] <= latency_constraint and
                    cur_config_perf["cost"] <= cost_constraint):
                if best_config is None:
                    best_config = cur_node_configs
                    best_config_perf = cur_config_perf
                    print("Initializing config to {} ({})".format(best_config, best_config_perf))
                else:
                    if cur_config_perf["throughput"] > best_config_perf["throughput"]:
                        best_config = cur_node_configs
                        best_config_perf = cur_config_perf
                        print("Updating config to {} ({})".format(best_config, best_config_perf))

        return best_config, best_config_perf


class GreedyOptimizer(object):

    def __init__(self, dag, scale_factors, node_profs):
        self.dag = dag
        self.scale_factors = scale_factors
        self.node_profs = node_profs

    def select_optimal_config(latency_constraint, cost_constraint):
        pass

    def _select_optimal_config_cloud(latency_constraint, cost_constraint, cloud):
        pass
