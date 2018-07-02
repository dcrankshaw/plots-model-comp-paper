import sys
import os
import json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath("../optimizer/"))
import single_node_profiles_cpp as snp
import utils
import matplotlib.pyplot as plt
import seaborn as sns

def plot():
    profs = snp.load_single_node_profiles(
            single_node_profs_dir=os.path.abspath("../results_cpp_benchmarker/contention_sweep"),
            models="all")
    idx = 0

    cmap = sns.hls_palette(8, l=.3, s=.8)
    name_map = {
            "inception": "Inception",
            "tf-kernel-svm": "Kernel SVM",
            "cascadepreprocess": "Image Pre-Process"
            }

    for model, p in profs.items():
        fig, (ax_thru, ax_lat) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        p = p.sort_values(["contention", "mean_batch_size"])
        markers = ['d', 'o', 'x', 's']
        iii = 0
        for b, g in p.groupby("mean_batch_size"):
            if b in [1, 8, 16, 32]:
                color = cmap[iii]
                marker = markers[iii]
                ax_thru.plot(g.contention, g.thru_stage_mean_throughput_qps, ms=8, marker=marker, label="Batch {}".format(int(b)), color=color)
                color = cmap[iii]
                ax_lat.plot(g.contention, g.p99_latency, marker=marker, ms=8, label="Batch {}".format(int(b)), color=color)
                iii += 1


        fs = 16
        ax_thru.set_xlabel("Background Load (QPS)", fontsize=fs)
        ax_thru.set_ylabel("Throughput (QPS)", fontsize=fs)
        ax_thru.set_ylim(bottom=0)
        ax_lat.set_ylim(bottom=0)
        ax_lat.set_xlabel("Background Load (QPS)", fontsize=fs)
        ax_lat.set_ylabel("P99 Latency (s)", fontsize=fs)
        ax_lat.xaxis.set_ticks(np.arange(0, 1001, 250))
        ax_thru.xaxis.set_ticks(np.arange(0, 1001, 250))

        # ax_thru.legend(ncol=2, loc=0)
        ax_lat.legend(ncol=2, loc=0)

        fig.suptitle(name_map[model], fontsize=20)
        plt.tight_layout(pad=3)
        base_dir = "contention_sweep"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        print(name_map[model])

        plt.savefig(os.path.join(base_dir, "{}.pdf".format(model)))

        idx += 1

if __name__ == '__main__':
    plot()
