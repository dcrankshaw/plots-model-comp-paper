import os
import shutil


def clean_old_results(results_dir):
    fs = os.listdir(results_dir)
    # sorted_fs = sorted(fs)
    splits = [f.split("-") for f in fs]
    dups_map = {}
    for name in splits:
        if len(name) <= 2:
            continue
        key = "-".join(name[:-1])
        val = name[-1]
        if key in dups_map:
            dups_map[key].append(val)
        else:
            dups_map[key] = [val, ]
    # print(dups_map)
    for dup in dups_map:
        if len(dups_map[dup]) > 1:
            for f in dups_map[dup][:-1]:
                # print(f)
                old_fname = os.path.join(results_dir, "-".join([dup, f]))
                new_fname = "{}.OLD".format(old_fname)
                shutil.move(old_fname, new_fname)

            # print("KEEP: {}".format("-".join([dup, dups_map[dup][-1]])))


def clean_aws_dups(results_dir):
    fs = os.listdir(results_dir)
    for f in fs:
        if "k80" in f:
            shutil.move(os.path.join(results_dir, f),
                        os.path.join(results_dir, "{}.OLD".format(f)))
    # sorted_fs = sorted(fs)
    splits = [f.split("-") for f in fs]
    dups_map = {}
    for name in splits:
        if len(name) <= 2:
            continue
        key = "-".join(name[:-1])
        val = name[-1]
        if key in dups_map:
            dups_map[key].append(val)
        else:
            dups_map[key] = [val, ]
    # print(dups_map)
    for dup in dups_map:
        if len(dups_map[dup]) > 1:
            for f in dups_map[dup][:-1]:
                # print(f)
                old_fname = os.path.join(results_dir, "-".join([dup, f]))
                new_fname = "{}.OLD".format(old_fname)
                shutil.move(old_fname, new_fname)


if __name__ == "__main__":
    clean_aws_dups(os.path.abspath(
        "../results_cpp_benchmarker/single_model_profs/tf-log-reg"))
    clean_aws_dups(os.path.abspath(
        "../results_cpp_benchmarker/single_model_profs/tf-kernel-svm"))
