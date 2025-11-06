import numpy as np
import pandas as pd


aminos = "ACDEFGHIKLMNPQRSTVWY-"


def get_matrix(positions, importance_list):
    position_residue_matrix = {pos:{a:0 for a in aminos} for pos in positions}
    for d in importance_list:
        f, v = d
        pos, res = f.split('_')
        position_residue_matrix[pos][res] = v

    return position_residue_matrix


def get_position_amino_count(positions, feature_table):
    position_residue_matrix = {pos:{a:0 for a in aminos} for pos in positions}

    for pos in positions:
        for res in feature_table[pos]:
            try:
                position_residue_matrix[pos][res] += 1
            except:
                pass

    return position_residue_matrix


def calc_sequence_entropy(counts):
    entropy = 0
    total = sum(counts.values())
    for aa, val in counts.items():
        if val > 0:
            entropy += -(val/total) * np.log2(val/total)
    return entropy


def make_importance_matrix(feature_table, residue_importance, class_name=None):
    positions = list(feature_table.columns[2:])
    # print(residue_importance)
    #position_residue_matrix = get_matrix(positions, residue_importance)
    position_count_matrix = get_position_amino_count(positions, feature_table)

    if class_name is None:
        outfile = f"msa_position_residue_matrix.tsv"
    else:
        outfile = f"msa_position_residue_matrix_{class_name}.tsv"

    with open(outfile, "w") as out:
        out.write("\t".join(map(str, [
            "msa_pos",
            "shapley",
            "\t".join(map(str, aminos)),
            "\t".join(map(str, aminos)),
            "entropy"
            ]))+"\n")

        for pos, counts in position_count_matrix.items():
            apos = int(pos.replace('p', ''))
            shapley = residue_importance[apos]["total"]
            entropy = calc_sequence_entropy(counts)
            out.write("\t".join(map(str, [
                apos,
                shapley,
                "\t".join(map(str, [residue_importance[apos][aa] for aa in aminos])),
                "\t".join(map(str, [counts[aa] for aa in aminos])),
                f"{entropy:.4f}"
            ]))+"\n")
