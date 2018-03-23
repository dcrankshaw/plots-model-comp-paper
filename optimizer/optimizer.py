import profiler
import itertools
import copy


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
                self.dag, self.scale_factors, cur_node_configs, self.node_profs)[0]
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

    def select_optimal_config(self, cloud, latency_constraint, cost_constraint, initial_config):
        cur_pipeline_config = initial_config
        if not profiler.is_valid_pipeline_config(cur_pipeline_config):
            print("ERROR: provided invalid initial pipeline configuration")
        iteration = 0
        while True:
            def try_upgrade_gpu(bottleneck):
                """
                Returns either the new config or False if the GPU could not be upgraded
                """
                new_bottleneck_config = copy.deepcopy(cur_pipeline_config[bottleneck])
                upgraded_gpu = upgrade_gpu(cloud, new_bottleneck_config.gpu_type)
                # print("upgraded gpu from {} to {}".format(new_bottleneck_config.gpu_type,
                #                                           upgraded_gpu))
                if upgraded_gpu:
                    new_bottleneck_config.gpu_type = upgraded_gpu
                    return new_bottleneck_config
                else:
                    return False

            def try_increase_batch_size(bottleneck):
                """
                Returns either the new config or False if the batch_size could not be increased
                """
                return self.node_profs[bottleneck].increase_batch_size(
                    cur_pipeline_config[bottleneck])

            def try_increase_replication_factor(bottleneck):
                """
                Returns the new config with an increased replication factor
                """
                new_bottleneck_config = copy.deepcopy(cur_pipeline_config[bottleneck])
                new_bottleneck_config.num_replicas += 1
                return new_bottleneck_config

            cur_estimated_perf, cur_bottleneck_node = profiler.estimate_pipeline_performance_for_config(
                    self.dag, self.scale_factors, cur_pipeline_config, self.node_profs)

            actions = {"gpu": try_upgrade_gpu,
                       "batch_size": try_increase_batch_size,
                       "replication_factor": try_increase_replication_factor
                       }

            best_action = None
            best_action_thru = None
            best_action_config = None

            for action in actions:
                new_bottleneck_config = actions[action](cur_bottleneck_node)
                # print(new_bottleneck_config)
                if new_bottleneck_config:
                    copied_pipeline_config = copy.deepcopy(cur_pipeline_config)
                    copied_pipeline_config[cur_bottleneck_node] = new_bottleneck_config
                    result = profiler.estimate_pipeline_performance_for_config(
                            self.dag, self.scale_factors, copied_pipeline_config, self.node_profs)
                    if result is not None:
                        new_estimated_perf, new_bottleneck_node = result
                        if (new_estimated_perf["latency"] <= latency_constraint and
                                new_estimated_perf["cost"] <= cost_constraint):
                            if new_estimated_perf["throughput"] < cur_estimated_perf["throughput"]:
                                print("Uh oh: monotonicity violated:\n Old config: {}\n New config: {}".format(
                                    cur_pipeline_config[cur_bottleneck_node],
                                    new_bottleneck_config
                                ))
                            # assert new_estimated_perf["throughput"] >= cur_estimated_perf["throughput"]
                            if best_action is None:
                                # print("Setting best_action to {} in iteration {}".format(action,
                                #                                                          iteration))
                                best_action = action
                                best_action_thru = new_estimated_perf["throughput"]
                                best_action_config = new_bottleneck_config
                            elif best_action_thru < new_estimated_perf["throughput"]:
                                # print("Setting best_action to {} in iteration {}".format(action,
                                #                                                          iteration))
                                best_action = action
                                best_action_thru = new_estimated_perf["throughput"]
                                best_action_config = new_bottleneck_config

            # No more steps can be taken
            if best_action is None:
                break
            else:
                cur_pipeline_config[cur_bottleneck_node] = best_action_config
                print("Upgrading bottleneck node {bottleneck} to {new_config}".format(
                    bottleneck=cur_bottleneck_node, new_config=best_action_config))
            iteration += 1

        # Finally, check that the selected profile meets the application constraints, in case
        # the user provided unsatisfiable application constraints
        cur_estimated_perf, _ = profiler.estimate_pipeline_performance_for_config(
                self.dag, self.scale_factors, cur_pipeline_config, self.node_profs)

        if (cur_estimated_perf["latency"] <= latency_constraint and
                cur_estimated_perf["cost"] <= cost_constraint):
            return cur_pipeline_config, cur_estimated_perf
        else:
            print("Error: No configurations found that satisfy application constraints")
            return False

def upgrade_gpu(cloud, cur_gpu):
    # gpu_rank = {"aws": ["none", "k80", "v100"],
    gpu_rank = {"aws": ["none", "v100"],
                "gcp": ["none", "k80", "p100"]
                }
    cloud_rank = gpu_rank[cloud]
    cur_gpu_idx = cloud_rank.index(cur_gpu)
    if cur_gpu_idx + 1 == len(cloud_rank):
        # print("Ran out of GPU upgrades")
        return False
    else:
        return cloud_rank[cur_gpu_idx + 1]
