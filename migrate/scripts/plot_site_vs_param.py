import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from migrate.make_feature_table import make_feature_table


def plot_site_vs_disc_param(data, pos):
    kpos = "p" + pos
    param_key = 'obj_param'

    cm = pd.crosstab(data[param_key], data[kpos])
    prot = "protein"

    print(cm)
    chi2, p, dof, expected = stats.chi2_contingency(cm)
    print(chi2, p)

    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.matshow(cm, cmap="Blues")

    plt.title(f"MSA position: {kpos}")
    #fig.colorbar(cax)

    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Label")

    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns)
    ax.set_yticklabels(cm.index)

    # 各セルに値を表示
    for (i, j), val in np.ndenumerate(cm.values):
        ax.text(j, i, f"{val}", ha="center", va="center", color="red")

    fig.savefig(f"{prot}_{kpos}.crosstab.pdf", bbox_inches="tight")


def plot_site_vs_cont_param(data, pos):
    kpos = "p"+pos
    param_key = 'obj_param'
    aminos = list(set(list(data[kpos])))
    data[param_key] = data[param_key].astype('float')
    df5 = {}
    mean_dict = {}
    median_dict = {}
    for aa in aminos:
        subset = data[data[kpos] == aa]
        n = subset.shape[0]
        #df5[aa + kpos[1:]] = list(subset[param_key])
        df5[aa] = list(subset[param_key])
        m = subset[param_key].mean()
        d = subset[param_key].median()
        s = subset[param_key].std()
        #mean_dict[aa + kpos[1:]] = m
        #median_dict[aa + kpos[1:]] = d
        mean_dict[aa] = m
        median_dict[aa] = d

    prot = "protein"
    # order by median values
    aa_order = []
    for aa, val in sorted(median_dict.items(), key=lambda x: x[1], reverse=True):
        aa_order.append(aa)

    sns.set(style='ticks', font_scale=1.5)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.despine(left=False)
    plt.title(f"MSA position: {kpos}")
    g = sns.stripplot(data=[df5[aa] for aa in aa_order])
    g = sns.boxplot(data=[df5[aa] for aa in aa_order], color='white',
                    whiskerprops={'visible': False}, showbox=False, showcaps=False, showfliers=False )
    g.set_xticks(range(len(aa_order)))
    #g.set_xticklabels(aa_order, rotation=45)
    g.set_xticklabels(aa_order)
    g.set_ylim(int(data[param_key].min()*0.9), int(data[param_key].max()*1.1))
    g.set_ylabel(r"Objective Paramter")
    fig.savefig(f"{prot}_{kpos}.medians.pdf", bbox_inches="tight")


def main():

    parser = argparse.ArgumentParser(description="MIGRATE: Machine learning-based Identification of Globally Adaptive Amino-acid Residues Associated with Tolerance to diverse Environments.")
    parser.add_argument("msa_file", help="MSA file")
    parser.add_argument("--msa-pos", help="MSA position")
    parser.add_argument("--param", default='cont', help="Parameter type [cont/disc], cont: continuous, disc: discrete")


    args = parser.parse_args()

    pos = args.msa_pos
    prm = args.param
    feature_table = make_feature_table(args.msa_file)

    print(feature_table.shape)
    if prm == 'disc':
        plot_site_vs_disc_param(feature_table, pos)
    else:
        plot_site_vs_cont_param(feature_table, pos)


if __name__ == "__main__":
    main()
