import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_residue_importance(df, pdb_id, pdb_start=0, class_name=None):
    sns.set(font_scale=1.6, style="ticks")
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.despine(left=False, bottom=False)
    ax.bar(df["residues"], df["values"],
           color="orangered", edgecolor="orangered", linewidth=0, width=0.8)
    ax.set_xlim(df["residues"].min()-0.5, df["residues"].max()+0.5)
    ax.set_ylabel("RI score")
    ax.set_xlabel("Residue number")
    ax.set_title(pdb_id)

    if class_name is not None:
        output_file = f"Importance_plot_{class_name}.{pdb_id}.pdf"
    else:
        output_file = f"Importance_plot_{pdb_id}.pdf"

    fig.savefig(output_file, bbox_inches="tight")


def map_importance(feature_table, residue_importance, class_name=None,
                   target_id=None, pdb_id=None, pdb_start=0):

    seq_table = feature_table[feature_table['seq_id'].isin([target_id])]
    seq_table = seq_table.drop(['seq_id', 'obj_param'], axis=1)

    pdb_chain = pdb_id[-1]
    pos = 0
    positions, t_seq = {}, ""
    i = 1
    for col, val in seq_table.items():
        msa_pos = i
        aa = val.values[0]
        if aa != '-':
            pos += 1
            t_seq += aa
            #print(p, pos, aa, col)
            try:
                positions[pos] = (residue_importance[msa_pos]['total'], msa_pos)
            except:
                positions[pos] = (0, msa_pos)

        i += 1

    residues, values = [], []
    k = 0

    if class_name is not None:
        tsv_file = f"residue_importance_{class_name}_{target_id}.tsv"
    else:
        tsv_file = f"residue_importance_{target_id}.tsv"

    if pdb_id is not None:
        if class_name is not None:
            cxc_file = f"residue_importance_{class_name}_{pdb_id}.cxc"
        else:
            cxc_file = f"residue_importance_{pdb_id}.cxc"

        with open(tsv_file, "w") as out:
            out.write("\t".join(map(str, [
                "msa_pos", "seqid", "pos", "pos_3d", "aa", "importance"
            ])) + "\n")
            for res, val in positions.items():
                imp, msa_pos = val
                res3d = res + pdb_start-1
                aa = t_seq[k]
                residues.append(res)
                values.append(imp)
                out.write("\t".join(map(str, [
                    msa_pos, target_id, res, res3d, aa, imp
                ])) + "\n")
                k += 1

        with open(cxc_file, "w") as out:
            for res, val in positions.items():
                imp, msa_pos = val
                res += pdb_start-1
                if res < 0:
                    continue
                out.write(" ".join(map(str, [
                    "setattr", f"/{pdb_chain}:{res}",
                    "residue score",
                    f"{imp:.5f}",
                    "create true"
                ])) + "\n")

        df = pd.DataFrame({"residues": residues, "values": values})
        plot_residue_importance(df, pdb_id, pdb_start=pdb_start, class_name=class_name)

    else:
        with open(tsv_file, "w") as out:
            out.write("\t".join(map(str, [
                "msa_pos", "seqid", "pos", "aa", "shapley"
            ])) + "\n")
            for res, val in positions.items():
                imp, msa_pos = val
                aa = t_seq[k]
                residues.append(res)
                values.append(imp)
                out.write("\t".join(map(str, [
                    msa_pos, target_id, res, aa, imp
                ])) + "\n")
                k += 1
