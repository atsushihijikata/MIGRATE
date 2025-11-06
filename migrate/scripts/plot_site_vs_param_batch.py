import argparse
import pandas as pd
import numpy as np
from scipy import stats
from migrate.make_feature_table import make_feature_table


def plot_site_vs_disc_param_batch(data):
    param_key = 'obj_param'
    msa_len = data.columns[2:]

    for kpos in msa_len:
        cm = pd.crosstab(data[param_key], data[kpos])

        try:
            chi2, p_val, dof, expected = stats.chi2_contingency(cm)
            if p_val < 0.05/len(msa_len):
                print("pos =", kpos, "chi2 =", chi2, "p =", p_val)
        except:
            continue


def plot_site_vs_cont_param_batch(data):
    param_key = 'obj_param'
    msa_len = data.columns[2:]
    aminos = "ACDEFGHIKLMNPQRSTVWY-"

    data[param_key] = data[param_key].astype(float)
    #print(data.columns)
    for kpos in msa_len:
        df5 = data[[param_key, kpos]]
        groups = [g[param_key].values for name, g in df5.groupby(kpos)]

        try:
            f_stat, p_val = stats.f_oneway(*groups)
            if p_val < 0.05/len(msa_len): # Bonferroni
                print("pos =", kpos, "F =", f_stat, "p =", p_val)
        except:
            continue


def plot_site_vs_cont_param(data, pos):
    kpos = "p"+pos
    param_key = 'obj_param'
    aminos = list(set(list(data[kpos])))
    df5 = {}
    mean_dict = {}
    median_dict = {}
    for aa in aminos:
        subset = data[data[kpos] == aa]
        n = subset.shape[0]
        df5[aa + kpos[1:]] = list(subset[param_key])
        m = subset[param_key].mean()
        d = subset[param_key].median()
        s = subset[param_key].std()
        mean_dict[aa + kpos[1:]] = m
        median_dict[aa + kpos[1:]] = d

    # order by median values
    aa_order = []
    for aa, val in sorted(median_dict.items(), key=lambda x: x[1], reverse=True):
        aa_order.append(aa)

    sns.set(style='ticks', font_scale=1.5)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.despine(left=False)
    g = sns.stripplot(data=[df5[aa] for aa in aa_order])
    g = sns.boxplot(data=[df5[aa] for aa in aa_order], color='white',
                    whiskerprops={'visible': False}, showbox=False, showcaps=False, showfliers=False )
    g.set_xticks(range(len(aa_order)))
    g.set_xticklabels(aa_order, rotation=45)
    g.set_ylim(0, 50)
    g.set_ylabel(r"Objective Paramter")
    fig.savefig(f"{outdir}/{prot}_{kpos}.medians.pdf", bbox_inches="tight")


def main():

    parser = argparse.ArgumentParser(description="MIGRATE: Machine learning-based Identification of Globally Adaptive Amino-acid Residues Associated with Tolerance to diverse Environments.")
    parser.add_argument("msa_file", help="MSA file")
    parser.add_argument("param_type", help="Parameter type", default="discrete")

    args = parser.parse_args()

    feature_table = make_feature_table(args.msa_file)
    ptype = args.param_type

    print(feature_table.shape)
    if ptype == 'discrete':
        plot_site_vs_disc_param_batch(feature_table)
    else:
        plot_site_vs_cont_param_batch(feature_table)


if __name__ == "__main__":
    main()