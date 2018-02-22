import os
import sys
import shutil



def clean_old_results(results_dir):
    fs = os.listdir(results_dir)
    # sorted_fs = sorted(fs)
    splits = [f.split("-") for f in fs]
    dups_map = {}
    for name in splits:
        if len(name) <= 2:
            continue
        key =  "-".join(name[:-1])
        val = name[-1]
        if key in dups_map:
            dups_map[key].append(val)
        else:
            dups_map[key] = [val,]
    # print(dups_map)
    for dup in dups_map:
        if len(dups_map[dup]) > 1:
            for f in dups_map[dup][:-1]:
                # print(f)
                old_fname = os.path.join(results_dir, "-".join([dup, f]))
                new_fname = "{}.OLD".format(old_fname)
                shutil.move(old_fname, new_fname)

            # print("KEEP: {}".format("-".join([dup, dups_map[dup][-1]])))


if __name__ == "__main__":
    clean_old_results("/Users/crankshaw/clipper-project/model-comp-project/plots-model-comp-paper/results/single_model_profs/pytorch-res50")
    # single_model_profs_dir=os.path.abspath("../results/single_model_profs/")
    # for m in os.listdir(single_model_profs_dir):
    #     fname = os.path.join(single_model_profs_dir, m)
    #     if os.path.isdir(fname):
    #         res = create_model_profile_df(fname)
