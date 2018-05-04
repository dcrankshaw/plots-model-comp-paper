
import os
import sys
import json
import time
from datetime import datetime
import numpy as np
from IPython.display import display, Markdown
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import seaborn as sns
import system_comparison_plots

def plot():

    df = pd.DataFrame(system_comparison_plots.load_all_inferline_sys_comp_results(
            "../results_cpp_benchmarker/e2e_no_netcalc/pipeline_one/latency_sensitivity"))

    colors = ["green", "magenta"]
    markers = ["o", "+"]

    # base_dir = "plots"
    base_dir = os.path.expanduser("~/Dropbox/Apps/ShareLaTeX/model-comp-paper/figs")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 1.5))
    idx = 0
    for slo, slo_group in df.groupby(df.slo):
        if slo == 0.3:
            slo_group = slo_group.sort_values("latency_percentage")
            ax.plot(100*(1.0 - slo_group.latency_percentage),
                    slo_group.slo_miss_rate,
                    marker=markers[idx],
                    c=colors[idx],
                    label="SLO {}".format(slo))
            idx += 1

    ax.set_ylabel("SLO Miss Rate", fontsize=8)
    ax.set_xlabel("% Latency Underestimate", fontsize=8)
    ax.set_ylim(0)
    ax.set_xlim(0)
    ax.tick_params(labelsize=8)
    ax.legend()

        # fig.suptitle("CV: {cv}, SLO: {slo}".format(cv=cv, slo=slo), fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(base_dir, 'ifl_pipe_one_latency_sensitivity.pdf'.format(slo)))

if __name__ == '__main__':
    plot()
