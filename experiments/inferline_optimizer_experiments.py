
import os
import sys
import json
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(cur_dir, "../optimizer")))
import single_node_profiles_cpp as snp
import profiler
import numpy as np
from optimizer import GreedyOptimizer
from utils import get_cpu_cost, get_gpu_cost
import logging
import math

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

MIN_DELAY_MS = 0.047


arrival_process_dir = os.path.join(cur_dir, "cached_arrival_processes")
if not os.path.exists(arrival_process_dir):
    os.makedirs(arrival_process_dir)

class Configuration(object):
    def __init__(self, slo, cost, lam, cv, node_configs, estimated_perf, response_time,
                 utilization):
        self.slo = slo
        self.cost = cost
        self.lam = lam
        self.node_configs = {n: c.__dict__ for n, c in node_configs.items()}
        self.estimated_perf = estimated_perf
        self.response_time = response_time
        self.cv = cv
        self.utilization = utilization


def generate_arrival_process(throughput, cv):
    arrival_process_dir = os.path.join(cur_dir, "cached_arrival_processes")
    if not os.path.exists(arrival_process_dir):
        raise Exception("Error: Cached arrival processes not found")

    def gamma(mean, CV, size):
        return np.random.gamma(1./CV, CV*mean, size=size)
    if cv == 1:
        deltas_path = os.path.join(arrival_process_dir,
                                   "{}.deltas".format(throughput))
    else:
        deltas_path = os.path.join(arrival_process_dir,
                                   "{lam}_{cv}.deltas".format(lam=throughput, cv=cv))
    if not os.path.exists(deltas_path):
        inter_request_delay_ms = 1.0 / float(throughput) * 1000.0
        deltas = gamma(inter_request_delay_ms, cv, size=50000)
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


def probe_throughputs(slo, cloud, cost, opt, cv, opt_function):
    min = 0
    max = 2000
    highest_successful_config = None
    while True:
        if max == min:
            break
        middle = min + math.ceil((max - min) / 2)
        logger.info("PROBING. min: {}, max: {}, middle: {}".format(
            min, max, middle))
        # result = optimize_pipeline_one(middle, opt, slo, cost, cloud, cv)
        result = opt_function(middle, opt, slo, cost, cloud, cv)
        if result:
            min = middle
            highest_successful_config = result
        else:
            max = middle - 1
    return (min, highest_successful_config)


########################################################
################### PIPELINE ONE (ID1) ################


def get_optimizer_pipeline_one(utilization):
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
            node_profs[name] = profiler.NodeProfile(name, profs[name], "latency_stage", utilization)
        else:
            node_profs[name] = profiler.NodeProfile(name, profs[name], "thru_stage", utilization)
    opt = GreedyOptimizer(dag, scale_factors, node_profs)
    return opt


def optimize_pipeline_one(throughput, opt, slo, cost, cloud, cv):
    arrival_history = generate_arrival_process(throughput, cv)
    results = []
    inception_gpu = "k80"
    resnet_gpu = "k80"
    if cloud == "aws":
        num_cpus = 1
        resnet_gpu = "v100"
    else:
        num_cpus = 2
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
    return result




def sweep_utilization_factor_pipeline_one():
    results_dir = os.path.abspath("utilization_sweep_pipeline_one")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info("Created results directory: %s" % results_dir)
    costs = [5.4, 10.6]
    cloud = "aws"
    logger.info("Optimizer initialized")
    cv = 1.0
    slo = 1.0
    for cost in costs:
        configs = []
        for utilization in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
            opt = get_optimizer_pipeline_one(utilization)
            results_fname = "aws_image_driver_one_utilization_sweep_slo_{slo}_cv_{cv}_cost_{cost}.json".format(
                slo=slo, cv=cv, cost=cost)
            results_file = os.path.join(results_dir, results_fname)
            lam, result = probe_throughputs(slo, cloud, cost, opt, cv, optimize_pipeline_one)
            if result:
                logger.info(("FOUND CONFIG FOR UTIL: {util}, COST: {cost}, LAMBDA: {lam}, "
                            "CV: {cv}").format(util=utilization, cost=cost, lam=lam, cv=cv))
                node_configs, perfs, response_time = result
                configs.append(Configuration(
                    slo, cost, lam, cv, node_configs, perfs,
                    response_time, utilization).__dict__)
                with open(results_file, "w") as f:
                    json.dump(configs, f, indent=4)
            else:
                logger.info("no result")


def generate_pipeline_one_configs(utilization=0.8):
    results_dir = os.path.abspath("e2e_sys_comp_pipeline_one/util_{}".format(utilization))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info("Created results directory: %s" % results_dir)
    costs = [5.4, 8.0, 10.6, 13.2, 15.8, 18.4, 21.0]
    cloud = "aws"
    opt = get_optimizer_pipeline_one(utilization)
    logger.info("Optimizer initialized")
    # for cv in [1.0, 4.0, 0.1]:
    for cv in [1.0, 4.0]:
        for slo in [0.5, 0.35, 1.0]:
            configs = []
            results_fname = "aws_image_driver_one_ifl_configs_slo_{slo}_cv_{cv}.json".format(
                slo=slo,
                cv=cv,
                util=utilization)
            results_file = os.path.join(results_dir, results_fname)
            for cost in costs:
                lam, result = probe_throughputs(slo, cloud, cost, opt, cv, optimize_pipeline_one)
                if result:
                    logger.info(("FOUND CONFIG FOR SLO: {slo}, COST: {cost}, LAMBDA: {lam}, "
                                "CV: {cv}").format(slo=slo, cost=cost, lam=lam, cv=cv))
                    node_configs, perfs, response_time = result
                    configs.append(Configuration(
                        slo, cost, lam, cv, node_configs, perfs,
                        response_time, utilization).__dict__)
                    with open(results_file, "w") as f:
                        json.dump(configs, f, indent=4)
                else:
                    logger.info("no result")

########################################################
########## PIPELINE THREE (Resnet Cascade) #############

def get_optimizer_pipeline_three(utilization):
    dag = profiler.get_logical_pipeline("pipeline_three")
    with open(os.path.abspath("../results_python_benchmarker/e2e_profs/systemx/resnet_cascade/"
                              "slo_500ms/alex_1-r50_1-r152_1-171025_083128.json")) as f:
        sample_run = json.load(f)
    scale_factors = profiler.get_node_scale_factors(sample_run, dag.reference_node)
    print("SCALE FACTORS: {}".format(scale_factors))
    node_configs = profiler.get_node_configs_from_experiment(sample_run)
    profs = snp.load_single_node_profiles(models=[n for n in node_configs])
    node_profs = {}
    for name in node_configs:
        node_profs[name] = profiler.NodeProfile(name, profs[name], "thru_stage", utilization)
    opt = GreedyOptimizer(dag, scale_factors, node_profs)
    return opt

def optimize_pipeline_three(throughput, opt, slo, cost, cloud, cv):
    arrival_history = generate_arrival_process(throughput, cv)
    results = []
    initial_gpu_type = "k80"
    if cloud == "aws":
        num_cpus = 1
    else:
        num_cpus = 2
    initial_config = {
        "res50": profiler.NodeConfig(name="res50",
                                     num_cpus=num_cpus,
                                     gpu_type=initial_gpu_type,
                                     batch_size=1,
                                     num_replicas=1,
                                     cloud=cloud),
        "res152": profiler.NodeConfig(name="res152",
                                      num_cpus=num_cpus,
                                      gpu_type=initial_gpu_type,
                                      batch_size=1,
                                      num_replicas=1,
                                      cloud=cloud),
        "alexnet": profiler.NodeConfig(name="alexnet",
                                      num_cpus=num_cpus,
                                      gpu_type=initial_gpu_type,
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
    return result

def sweep_utilization_factor_pipeline_three():
    results_dir = os.path.abspath("utilization_sweep_pipeline_three")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info("Created results directory: %s" % results_dir)
    costs =[11.08]
    cloud = "aws"
    logger.info("Optimizer initialized")
    cv = 1.0
    slo = 1.0
    for cost in costs:
        configs = []
        for utilization in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
            opt = get_optimizer_pipeline_three(utilization)
            results_fname = "aws_resnet_cascade_utilization_sweep_slo_{slo}_cv_{cv}_cost_{cost}.json".format(
                slo=slo, cv=cv, cost=cost)
            results_file = os.path.join(results_dir, results_fname)
            lam, result = probe_throughputs(slo, cloud, cost, opt, cv, optimize_pipeline_three)
            if result:
                logger.info(("FOUND CONFIG FOR UTIL: {util}, COST: {cost}, LAMBDA: {lam}, "
                            "CV: {cv}").format(util=utilization, cost=cost, lam=lam, cv=cv))
                node_configs, perfs, response_time = result
                configs.append(Configuration(
                    slo, cost, lam, cv, node_configs, perfs,
                    response_time, utilization).__dict__)
                with open(results_file, "w") as f:
                    json.dump(configs, f, indent=4)
            else:
                logger.info("no result")

def generate_pipeline_three_configs(utilization=0.7):
    results_dir = os.path.abspath("e2e_sys_comp_pipeline_three/util_{}".format(utilization))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info("Created results directory: %s" % results_dir)
    cloud = "aws"
    cost_lower_bound = get_cpu_cost(cloud, 3) + get_gpu_cost(cloud, "v100", 2) + get_gpu_cost(cloud, "k80", 1)
    cost_upper_bound = get_cpu_cost(cloud, 16) + get_gpu_cost(cloud, "v100", 8) + get_gpu_cost(cloud, "k80", 8)
    cost_increment = get_cpu_cost(cloud, 1) + get_gpu_cost(cloud, "v100", 1)
    print(cost_lower_bound, cost_upper_bound, cost_increment)
    costs = np.arange(cost_lower_bound, cost_upper_bound+1.0, cost_increment)
    print(costs)
    cloud = "aws"
    opt = get_optimizer_pipeline_three(utilization)
    logger.info("Optimizer initialized")
    # for cv in [1.0, 4.0, 0.1]:
    for cv in [4.0, 0.1]:
        for slo in [0.5, 0.35, 1.0]:
            configs = []
            results_fname = "aws_resnet_cascade_ifl_configs_slo_{slo}_cv_{cv}.json".format(
                slo=slo,
                cv=cv,
                util=utilization)
            results_file = os.path.join(results_dir, results_fname)
            for cost in costs:
                lam, result = probe_throughputs(slo, cloud, cost, opt, cv, optimize_pipeline_three)
                if result:
                    logger.info(("FOUND CONFIG FOR SLO: {slo}, COST: {cost}, LAMBDA: {lam}, "
                                "CV: {cv}").format(slo=slo, cost=cost, lam=lam, cv=cv))
                    node_configs, perfs, response_time = result
                    configs.append(Configuration(
                        slo, cost, lam, cv, node_configs, perfs,
                        response_time, utilization).__dict__)
                    with open(results_file, "w") as f:
                        json.dump(configs, f, indent=4)
                else:
                    logger.info("no result")

# def debug_pipeline_three():
#     cost = 11.08
#     cloud = "aws"
#     logger.info("Optimizer initialized")
#     cv = 1.0
#     slo = 1.0
#     utilization = 1.0
#     lam = 200
#     opt = get_optimizer_pipeline_three(utilization)
#     result = optimize_pipeline_three(lam, opt, slo, cost, cloud, cv)
#     if result:
#         logger.info(("FOUND CONFIG FOR UTIL: {util}, COST: {cost}, LAMBDA: {lam}, "
#                     "CV: {cv}").format(util=utilization, cost=cost, lam=lam, cv=cv))
#         node_configs, perfs, response_time = result
#         config_res = Configuration(
#             slo, cost, lam, cv, node_configs, perfs,
#             response_time, utilization).__dict__
#         logger.info("Result:\n{}".format(json.dumps(config_res, indent=2)))
#     else:
#         logger.info("no result")

if __name__ == "__main__":
    # generate_pipeline_one_configs(0.7)
    # generate_pipeline_three_configs(0.7)
    sweep_utilization_factor_pipeline_three()
    # debug_pipeline_three()
