import os
import json
import profiler
import single_node_profiles as snp
from optimizer import BruteForceOptimizer


def cascades_example(cloud, latency_constraint, cost_constraint):
    # Load the single node profiles
    profs = snp.load_single_node_profiles()

    # Get the hardcoded logical DAG structure. In the real system, this would
    # be extracted using lineage from the driver.
    dag = profiler.get_logical_pipeline("pipeline_three")

    # Load the scale factors from an example run of the pipeline
    results_path = ("../results/e2e_profs/systemx/resnet_cascade/slo_500ms/"
                    "alex_1-r50_1-r152_2-171025_083730.json")
    with open(os.path.abspath(results_path)) as f:
        sample_run = json.load(f)
    scale_factors = profiler.get_node_scale_factors(sample_run, dag.reference_node)
    node_profs = {name: profiler.NodeProfile(name, profs[name]) for name in dag.nodes()}

    # Choose optimizer implementation
    opt = BruteForceOptimizer(dag, scale_factors, node_profs)

    # Cloud can be either "aws" or "gcp". Latency constraint is in seconds,
    # cost constraint is in $/hr.
    return opt.select_optimal_config(cloud, latency_constraint, cost_constraint)


if __name__ == "__main__":
    print(cascades_example("aws", 0.7, 8))
