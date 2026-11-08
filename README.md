# MIGRATE

**M**achine learning-based **I**dentification of **G**lobally adaptive amino-acid **R**esidues
**A**ssociated with functional **T**raits and diverse **E**nvironmental factors

---

## Prerequisites

python 3.9 or later.



## Installation

```bash

tar zxf migrate_package.tgz

cd migrate_package

pip install .

```

## Usage

```bash

migrate msa_file --target seq_id [--mode classification] [--model rf|xgb] [--n-estimators N] [--n-jobs N] [--explain shap|importance] [--cv-group group_file] [--cv-ri]

```

### Example

```bash
migrate migrate/data/Myb_cls.fas --target Msp.Q0KIY5.3.MYG.KOGBR.pygmy.sperm.whale \
                                 --pdbid 6bmgA --pdb-start 0 \
                                 --mode classification \
                                 --seed 123 

migrate migrate/data/GFP_rgr.fas --target mKalama1 \
                                 --pdbid 4ornB \
                                 --pdb-start 0 \
                                 --mode regression \
                                 --seed 123 

```

### Number of trees (`--n-estimators`)

`--n-estimators` sets the number of trees: default 1000 for RandomForest (`--model rf`)
and 100 for XGBoost (`--model xgb`). The value used is written to `cv_results.tsv`.
With more trees, RI scores of RandomForest depend less on `--seed`; see `analysis/rf_seed_stability.py`.

### Number of CPUs (`--n-jobs`)

`--n-jobs` sets the number of CPUs used for training the models and computing SHAP values (`-1`: all CPUs).
By default, RandomForest uses 1 CPU and XGBoost uses all CPUs.
Results do not depend on the number of CPUs (RandomForest predictions may differ at the level of floating-point rounding).

In classification mode, class imbalance is corrected by weighting each class inversely proportional to its frequency
in the training data (n_samples / (n_classes × n_samples_of_class)):
RandomForest uses `class_weight="balanced"`, and XGBoost uses the same weights as `sample_weight`.
The setting is written to `cv_results.tsv` (`class_weight`).

### Residue importance score (`--explain`)

The RI (residue importance) score of each MSA position is computed from the model trained on all sequences.
Each MSA position is one-hot encoded by amino acid (e.g., `p28_D`, `p28_E`),
and the RI score of a position is the sum over its amino-acid features, normalized so that the sum over all positions is 1.

| `--explain` | Feature importance | Note |
|---|---|---|
| `shap` (default) | Mean absolute SHAP value of each feature (`shap.TreeExplainer`, interventional) | Takes longer |
| `importance` | Model's built-in `feature_importances_`: mean decrease in impurity for RandomForest, gain for XGBoost | Fast; impurity-based importance may be biased among correlated features |

For multi-class classification with `shap`, RI scores are computed for each class, and the maximum over classes at each position is also reported (`maxshap`).
With `--cv-ri`, RI scores and their ranks computed from the model of each CV fold are also written to `cv_ri_scores.tsv`
(columns `RI_<fold>` and `rank_<fold>`; `<fold>` is the group with `--cv-group`, otherwise the fold number),
and the mean and 95% CI of Pearson's r and Spearman's rho between the RI of the full model and of each fold
are added to `cv_results.tsv`. This computes RI scores once per fold and takes time.

#### Position-residue matrix (`msa_position_residue_matrix.tsv`)

One row per MSA position (`msa_position_residue_matrix_<class>.tsv` for each class in multi-class classification).

| Column | Description |
|---|---|
| `msa_pos` | MSA position (1-based) |
| `RI_score` | RI score of the position |
| `A` ... `-` (1st set) | Raw importance of each amino-acid feature at the position |
| `A` ... `-` (2nd set) | Number of sequences with each amino acid (or gap) at the position |
| `entropy` | Shannon entropy (bits) of the residues at the position (gaps counted as a state) |
| `mutual_info` | Mutual information (bits) between the residues at the position and the class label (gaps counted as a state). Classification only |
| `mutual_info_z` | Z-score of `mutual_info` against the null distribution obtained by permuting the class labels (100 permutations). Classification only |

`mutual_info` and `mutual_info_z` are output only in classification mode (not in regression mode).
`mutual_info` is computed from the empirical joint distribution of residues and class labels.
Mutual information is biased upward for small datasets and for positions with many residue types;
`mutual_info_z` = (MI − mean of null MI) / SD of null MI corrects for this bias (random seed: `--seed`).
`mutual_info_z` is `nan` when MI does not vary under permutation (e.g., invariant positions).
Note that the permutation test assumes independent sequences, so Z-scores may be inflated by phylogenetic relatedness.

### MSA data format

A multiple sequence alignment in FASTA format with target parameters in each header.
The header must have exactly two fields separated by a space: sequence ID and target parameter
(class label for classification, numerical value for regression).
Do not add other fields (e.g., CV group) to the header; such sequences are skipped.

\>seqid1 param1  
SEQUENCE-ALIGNMENT...  
\>seqid2 param2  
SEQUENCE-ALIGNMENT...  
...

### CV group file format (optional, `--cv-group`)

By default, leave-one-out (< 60 sequences) or 10-fold CV is performed.
In all CV modes, per-sequence predictions (and class probabilities for classification) are written to `cv_predictions.tsv`
(column `fold`: fold number, or column `group` with `--cv-group`), and per-fold RI scores to `cv_ri_scores.tsv` (with `--cv-ri`).
CV performance metrics shown in the log are saved to `cv_results.tsv` (`metric<TAB>value`):
Pearson's r, Spearman's rho (and its p-value), R2 and RMSE relative to the baseline (mean of training fold), and RMSE for regression;
confusion matrix (one row per true/predicted class pair), ROC-AUC, accuracy, MCC and macro F1 for classification.
With `--cv-group` (alias: `--cv-fold`), leave-one-group-out CV is performed:
each group is used once as validation data, and the other sequences are used for training.

A tab- or space-separated file of sequence ID and group.

```
seqid1	1
seqid2	1
seqid3	2
seqid4	train_only
...
```

- Group is a CV group (e.g., clade) such as `1`, `2`, ..., **not a class label**. Class labels and target values are read from the MSA header only.
- `train_only`: always used for training, never used for validation. Sequences not listed in this file are also treated as `train_only`.
- Sequence IDs must exactly match the first field of the MSA header (e.g., `NP_001276998` and `NP.001276998` are different IDs).
- Lines starting with `#` are ignored.

The number of sequences per group and label is printed at the start of CV.
Check that each class is included in the training data of every group.

```bash
migrate migrate/data/Myb_cls.fas --target Msp.Q0KIY5.3.MYG.KOGBR.pygmy.sperm.whale \
                                 --mode classification \
                                 --cv-group groups.tsv
```
