import numpy as np
import pandas as pd


aminos = "ACDEFGHIKLMNPQRSTVWY-"
n_permutations = 100  # for Z-score of mutual information


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


def calc_mutual_information(feature_table, positions, seed=0):
    # mutual information (bits) between residues at each MSA position and the class label,
    # and its Z-score against the null distribution obtained by permuting the class labels
    y = feature_table['obj_param'].values
    X = np.array([pd.factorize(feature_table[pos])[0] for pos in positions]).T
    n, n_pos = X.shape
    n_res = X.max() + 1
    y_codes, y_uniques = pd.factorize(y)
    n_cls = len(y_uniques)
    # index of (position, residue) for each sequence; joint counts are obtained by one bincount
    offset = (np.arange(n_pos) * n_res * n_cls)[None, :] + X * n_cls

    def _mi(target):
        # same as sklearn.metrics.mutual_info_score for all positions at once (bits)
        joint = np.bincount((offset + target[:, None]).ravel(),
                            minlength=n_pos * n_res * n_cls).reshape(n_pos, n_res, n_cls) / n
        px = joint.sum(axis=2, keepdims=True)
        py = joint.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = np.where(joint > 0, joint * np.log2(joint / (px * py)), 0.0)
        return terms.sum(axis=(1, 2))

    y = y_codes

    mi = _mi(y)
    rng = np.random.default_rng(seed)
    null = np.array([_mi(rng.permutation(y)) for _ in range(n_permutations)])
    null_sd = null.std(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        # NaN when MI does not vary under permutation (e.g., invariant positions)
        mi_z = np.where(null_sd > 1e-10, (mi - null.mean(axis=0)) / null_sd, np.nan)
    return dict(zip(positions, mi)), dict(zip(positions, mi_z))


def make_importance_matrix(feature_table, residue_importance, class_name=None, mode='classification', seed=0):
    positions = list(feature_table.columns[2:])
    with_mi = mode == 'classification'  # mutual information is not computed for regression
    if with_mi:
        mutual_info, mutual_info_z = calc_mutual_information(feature_table, positions, seed=seed)
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
            "RI_score",
            "\t".join(map(str, aminos)),
            "\t".join(map(str, aminos)),
            "entropy",
            *(["mutual_info", "mutual_info_z"] if with_mi else [])
            ]))+"\n")

        for pos, counts in position_count_matrix.items():
            apos = int(pos.replace('p', ''))
            ri_score = residue_importance[apos]["total"]
            entropy = calc_sequence_entropy(counts)
            out.write("\t".join(map(str, [
                apos,
                ri_score,
                "\t".join(map(str, [residue_importance[apos][aa] for aa in aminos])),
                "\t".join(map(str, [counts[aa] for aa in aminos])),
                f"{entropy:.4f}",
                *([f"{mutual_info[pos]:.4f}", f"{mutual_info_z[pos]:.4f}"] if with_mi else [])
            ]))+"\n")
