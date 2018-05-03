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

spd_pipeline_one = system_comparison_plots.load_spd_pipeline_one()
ifl_pipeline_one = system_comparison_plots.load_inferline_pipeline_one()

print len(spd_pipeline_one), len(ifl_pipeline_one)

pipeline_one_df = pd.concat([spd_pipeline_one, ifl_pipeline_one])

names = pipeline_one_df.name.unique()
colors = sns.color_palette(n_colors=len(names))
cmap = dict(zip(names, colors))
print cmap
cmap = {
    "SPD-mean_provision":"red",
    "InferLine":"blue"
}
label_map = {
    "SPD-mean_provision":"SPD",
    "InferLine":"IFL"
}
marker_map = {
    "SPD-mean_provision":"^",
    "InferLine":"o"
}

for cv, cv_group in pipeline_one_df.groupby(pipeline_one_df.CV):
    if cv not in [0.1, 4.0]:
        continue
    for slo, slo_group in cv_group.groupby(cv_group.slo):
        fig, (ax_cost) = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
        ng = slo_group.groupby(slo_group.name)
        ifl_costs = ng.get_group("InferLine").sort_values("lambda")
        for name, name_group in ng:
            name_group = name_group.sort_values("lambda")
            ax_cost.step(name_group["lambda"], name_group.throughput/name_group.cost, "-", where="post", c=cmap[name], label=label_map[name])
            ax_cost.scatter(name_group["lambda"], name_group.throughput/name_group.cost, marker=marker_map[name], c=cmap[name])

        ax_cost.set_ylabel("QPSD", fontsize=13)
        ax_cost.set_xlabel("Throughput (QPS)", fontsize=13)
        ax_cost.set_ylim(0)
        ax_cost.set_xlim(0)
        ax_cost.tick_params(labelsize=13)
        ax_cost.legend()

        fig.savefig('CV_{}_SLO_{}.pdf'.format(cv,slo))

        fig.suptitle("CV: {cv}, SLO: {slo}".format(cv=cv, slo=slo), fontsize=14)
        plt.tight_layout(pad=3)

