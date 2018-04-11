
import os
import sys
import json
# import time
# from datetime import datetime
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(cur_dir, "../optimizer")))
import single_node_profiles_cpp as snp
import profiler
# import end_to_end_profiles as e2e_profs
import numpy as np
from optimizer import GreedyOptimizer
# import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

MIN_DELAY_MS = 0.047


arrival_process_dir = os.path.join(cur_dir, "cached_arrival_processes")
if not os.path.exists(arrival_process_dir):
    os.makedirs(arrival_process_dir)


def generate_arrival_process(throughput, cv=1):
    if cv == 1:
        deltas_path = os.path.join(arrival_process_dir,
                                   "{}.deltas".format(throughput))
        if not os.path.exists(deltas_path):
            inter_request_delay_ms = 1.0 / float(throughput) * 1000.0
            deltas = np.random.exponential(inter_request_delay_ms, size=(50000))
            deltas = np.clip(deltas, a_min=MIN_DELAY_MS, a_max=None)
            arrival_history = np.cumsum(deltas)
            with open(deltas_path, "w") as f:
                for d in deltas:
                    f.write("{}\n".format(d))
        else:
            with open(deltas_path, "r") as f:
                deltas = np.array([float(l.strip()) for l in f]).flatten()
            arrival_history = np.cumsum(deltas)
        return arrival_history
    else:
        raise Exception("Eyal needs to implement this")


def get_optimizer_pipeline_one():
    dag = profiler.get_logical_pipeline("pipeline_one")
    with open(os.path.abspath("../results_python_benchmarker/e2e_profs/systemx/image_driver_1/500ms"
                              "/incep_1-logreg_1-ksvm_1-resnet_1-171221_091209.json")) as f:
        sample_run = json.load(f)
    scale_factors = profiler.get_node_scale_factors(sample_run, dag.reference_node)
    node_configs = profiler.get_node_configs_from_experiment(sample_run)
    profs = snp.load_single_node_profiles(models=[n for n in node_configs])
    node_profs = {}
    for name in node_configs:
        if name in ["tf-log-reg", "tf-kernel-svm"]:
            node_profs[name] = profiler.NodeProfile(name, profs[name], "latency_stage")
        else:
            node_profs[name] = profiler.NodeProfile(name, profs[name], "thru_stage")
    opt = GreedyOptimizer(dag, scale_factors, node_profs)
    return opt


def rerun_config():
    slo = 0.25
    cost = 5.4
    cloud = "aws"
    cv = 1
    throughput = 121
    opt = get_optimizer_pipeline_one()

    arrival_history = generate_arrival_process(throughput, cv)

    results = []
    inception_gpu = "k80"
    num_cpus = 1
    resnet_gpu = "v100"
    initial_config = {
        "inception": profiler.NodeConfig(name="inception",
                                         num_cpus=num_cpus,
                                         gpu_type=inception_gpu,
                                         batch_size=1,
                                         num_replicas=1,
                                         cloud=cloud),
        "tf-resnet-feats": profiler.NodeConfig(name="tf-resnet-feats",
                                               num_cpus=num_cpus,
                                               gpu_type=resnet_gpu,
                                               batch_size=1,
                                               num_replicas=1,
                                               cloud=cloud),
        "tf-log-reg": profiler.NodeConfig(name="tf-log-reg",
                                          num_cpus=num_cpus,
                                          gpu_type="none",
                                          batch_size=1,
                                          num_replicas=1,
                                          cloud=cloud),
        "tf-kernel-svm": profiler.NodeConfig(name="tf-kernel-svm",
                                             num_cpus=num_cpus,
                                             gpu_type="none",
                                             batch_size=1,
                                             num_replicas=1,
                                             cloud=cloud),
    }
    result = opt.select_optimal_config(
        cloud, latency_constraint=slo, cost_constraint=cost, initial_config=initial_config,
        arrival_history=arrival_history, use_netcalc=True)
    if result:
        results.append(result)
        best_config, best_config_perf, response_time = result
        for b in best_config.items():
            print(b)
        print("\n\nFINAL RESULTS:")
        for r in results:
            print(r)
    return result


def estimate_per_node_perf():
    cloud = "aws"
    num_cpus = 1
    inception_gpu = "v100"
    resnet_gpu = "v100"

    opt = get_optimizer_pipeline_one()

    config = {
        "inception": profiler.NodeConfig(name="inception",
                                         num_cpus=num_cpus,
                                         gpu_type=inception_gpu,
                                         batch_size=16,
                                         num_replicas=1,
                                         cloud=cloud),
        "tf-resnet-feats": profiler.NodeConfig(name="tf-resnet-feats",
                                               num_cpus=num_cpus,
                                               gpu_type=resnet_gpu,
                                               batch_size=32,
                                               num_replicas=1,
                                               cloud=cloud),
        "tf-log-reg": profiler.NodeConfig(name="tf-log-reg",
                                          num_cpus=num_cpus,
                                          gpu_type="none",
                                          batch_size=1,
                                          num_replicas=1,
                                          cloud=cloud),
        "tf-kernel-svm": profiler.NodeConfig(name="tf-kernel-svm",
                                             num_cpus=num_cpus,
                                             gpu_type="none",
                                             batch_size=4,
                                             num_replicas=1,
                                             cloud=cloud),
    }

    for n in opt.node_profs:
        perf = opt.node_profs[n].estimate_performance(config[n])
        print("NODE: {}, PERF: {}".format(n, perf))


if __name__ == "__main__":
    rerun_config()
    # generate_pipeline_one_configs()
    # estimate_per_node_perf()
