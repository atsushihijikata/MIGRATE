import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from collections import Counter
import json
from scipy.stats import kruskal, spearmanr
from scipy import stats
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay, mean_squared_error
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from joblib import Parallel, delayed
from .make_matrix import aminos


default_seed = 123
n_estimators_rf = 1000  # default number of trees of RandomForest
n_estimators_xgb = 100  # default number of trees of XGBoost


def fit_model(model, X, y):
    # XGBClassifier has no class_weight; balance classes by sample weights
    # (n_samples / (n_classes * n_samples_of_class), same as class_weight="balanced" of RandomForest)
    if isinstance(model, xgb.XGBClassifier):
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)


def plot_confusion_matrix(y_true, y_pred, classes, class_names):
    # classes: sorted class values in y; class_names: their label names (same order)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    print("Confusion Matrix (rows: true, columns: predicted)")
    print(pd.DataFrame(cm, index=[f"true={c}" for c in class_names],
                       columns=[f"pred={c}" for c in class_names]).to_string())

    plt.savefig("classification_confusion_matrix.pdf", bbox_inches="tight")


def plot_regression_scatter(y_true, y_pred):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

    g = sns.jointplot(x='y_true', y='y_pred', data=df, kind='reg',
                      scatter_kws=dict(s=5), line_kws=dict(color='orangered')
                      )
    g.set_axis_labels("Target value", "Predicted value")
    g.savefig("regression_scatter.pdf", bbox_inches="tight")


def plot_shap_summary(shap_values, X, mode='classification', topn=10):

    plt.figure(figsize=(7, 7))

    ax = plt.gca()
    fig = plt.gcf()

    # Set background color to white
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    if mode == 'classification':
        if shap_values.ndim == 2:
            shap.summary_plot(shap_values, X, max_display=topn, show=False)
        else:
            shap.summary_plot(shap_values[:, :, 1], X, max_display=topn, show=False)
    else:
        shap.summary_plot(shap_values, X, max_display=topn, show=False)

    plt.savefig("shap_summary.pdf", format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()


def read_cv_folds(fold_file):
    # Read "seq_id<TAB>group" file. group is 1, 2, ... or "train_only".
    folds = {}
    for line in open(fold_file):
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        sid, fold = line.split()[:2]
        folds[sid] = fold
    return folds


def cross_valid(model, X, y, mode=None, seed=default_seed, folds=None, ids=None, le=None,
                feature_names=None, explain="shap", cv_ri=False):
    # cross-validation

    if folds is not None:  # leave-one-group-out
        groups = np.array([folds.get(i, 'train_only') for i in ids])
        missing = [i for i in ids if i not in folds]
        if missing:
            sys.stderr.write(f"Warning: {len(missing)} sequences not in cv-group file are used as train_only.\n")
        unmatched = [i for i, f in folds.items() if f != 'train_only' and i not in set(ids)]
        if unmatched:
            sys.stderr.write(f"Warning: {len(unmatched)} test sequences in cv-group file are not found in MSA: {', '.join(unmatched)}\n")
        test_folds = sorted(set(groups) - {'train_only'}, key=lambda f: (not f.isdigit(), int(f) if f.isdigit() else f))
        if not test_folds:
            sys.exit("Error: no test group found in cv-group file.")
        sys.stderr.write(f"Leave-one-group-out CV: {len(test_folds)} groups\n")
        splits = [(np.where(groups != f)[0], np.where(groups == f)[0]) for f in test_folds]
        cv_method = f"leave-one-group-out ({len(test_folds)} groups)"
        # summary of labels in each group
        labels = le.inverse_transform(y) if le is not None else y
        if mode == 'classification':
            summary = pd.crosstab(pd.Series(groups, name='group'), pd.Series(labels, name='label'))
        else:
            summary = pd.Series(labels.astype(float)).groupby(pd.Series(groups, name='group')).agg(['count', 'mean', 'min', 'max'])
        summary = summary.reindex([f for f in test_folds + ['train_only'] if f in summary.index])
        print(f"Number of sequences per group:\n{summary}")
    elif X.shape[0] < 60:  # for small datasets
        splits = LeaveOneOut().split(X)
        cv_method = "leave-one-out"
    else:
        splits = KFold(n_splits=10, shuffle=True, random_state=seed).split(X)
        cv_method = "10-fold"

    y_true, y_pred, test_ids, y_prob = [], [], [], []
    test_folds_of_ids = []  # fold (group) label of each test sequence
    y_base = []  # baseline: mean of training targets in each fold
    fold_ri = {}  # per-fold RI scores {class: {group: Series(msa_pos -> RI)}}
    train_f1s, valid_f1s = [], []
    classes = np.unique(y)

    for k_fold, (train_index, test_index) in enumerate(splits, 1):
        # fold label: group name for leave-one-group-out, fold number (1, 2, ...) for 10-fold CV / LOO
        fold_label = groups[test_index[0]] if folds is not None else str(k_fold)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        fit_model(model, X_train, y_train)

        if X_test.shape[0] == 1:
            y_pred.append(model.predict(X_test)[0])
            y_true.append(y_test[0])
        else:
            y_pred.extend(model.predict(X_test))
            y_true.extend(y_test)
        if ids is not None:
            test_ids.extend(ids[test_index])
            test_folds_of_ids.extend([fold_label] * len(test_index))
        if mode == 'regression':
            y_base.extend([np.mean(y_train)] * len(test_index))
        if cv_ri and feature_names is not None:  # per-fold RI only with --cv-ri (time-consuming)
            # RI scores of the model trained in this fold
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
            if explain == "shap":
                ri = per_residue_shapley(model, X_train_df, mode=mode, plot=False)
            else:
                ri = per_residue_importance(model, X_train_df, mode=mode)
            for cls, scores in ri.items():
                fold_ri.setdefault(cls, {})[fold_label] = pd.Series({pos: v['total'] for pos, v in scores.items()})
        if mode == 'classification':
            # align probability columns to all classes (a class may be absent in training)
            prob = np.zeros((X_test.shape[0], len(classes)))
            prob[:, np.searchsorted(classes, model.classes_)] = model.predict_proba(X_test)
            y_prob.extend(prob)
            # per-fold macro F1 on training and validation data
            train_f1s.append(f1_score(y_train, model.predict(X_train), average="macro", zero_division=0))
            valid_f1s.append(f1_score(y_test, model.predict(X_test), average="macro", zero_division=0))
            if folds is not None:
                print(f"Group {groups[test_index[0]]}: train F1 = {train_f1s[-1]}, validation F1 = {valid_f1s[-1]}")

    if ids is not None:  # output per-sequence prediction
        out_true, out_pred = (le.inverse_transform(y_true), le.inverse_transform(y_pred)) if le is not None else (y_true, y_pred)
        out = pd.DataFrame({'seq_id': test_ids, 'group' if folds is not None else 'fold': test_folds_of_ids,
                            'y_true': out_true, 'y_pred': out_pred})
        if mode == 'classification':
            class_names = le.inverse_transform(classes) if le is not None else classes
            for k, c in enumerate(class_names):
                out[f"prob_{c}"] = np.array(y_prob)[:, k]
        out.to_csv("cv_predictions.tsv", sep='\t', index=False)
        sys.stderr.write("Write per-sequence CV predictions to cv_predictions.tsv\n")

    for cls, group_ri in fold_ri.items():  # output per-fold RI scores and ranks
        ri_table = pd.DataFrame(index=sorted(set().union(*[s.index for s in group_ri.values()])))
        ri_table.index.name = 'msa_pos'
        for g, ri in group_ri.items():
            ri_table[f"RI_{g}"] = ri
            ri_table[f"rank_{g}"] = ri.rank(ascending=False, method='min').astype(int)
        if len(fold_ri) == 1:
            ri_file = "cv_ri_scores.tsv"
        else:
            cls_name = le.inverse_transform([cls])[0] if le is not None and cls != "maxshap" else cls
            ri_file = f"cv_ri_scores_{cls_name}.tsv"
        ri_table.to_csv(ri_file, sep='\t')
        sys.stderr.write(f"Write per-fold RI scores and ranks to {ri_file}\n")

    if mode == 'regression':
        plot_regression_scatter(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        rho, p_value = spearmanr(y_true, y_pred)
        print(f"corrcoef: {corr}")
        print(f"rho: {rho}, {p_value}")
        # R2 relative to the baseline (mean of training fold), not the mean of all test data
        mse_base = mean_squared_error(y_true, y_base)
        print(f"R2 (baseline: mean of training fold): {1 - mse / mse_base}")
        print(f"Baseline RMSE (mean of training fold): {np.sqrt(mse_base)}")
        write_cv_results([
            ("cv_method", cv_method), ("n_estimators", model.get_params().get('n_estimators')),
            ("n_sequences", len(y_true)),
            ("pearson_r", corr), ("spearman_rho", rho), ("spearman_p", p_value),
            ("R2 (baseline: mean of training fold)", 1 - mse / mse_base),
            ("baseline_RMSE (mean of training fold)", np.sqrt(mse_base)), ("RMSE", np.sqrt(mse)),
        ])
        return np.sqrt(mse), fold_ri

    elif mode == 'classification':
        class_names = le.inverse_transform(classes) if le is not None else classes
        plot_confusion_matrix(y_true, y_pred, classes, class_names)
        y_prob = np.array(y_prob)
        try:
            if len(classes) == 2:
                auc = roc_auc_score(np.array(y_true) == classes[1], y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr', labels=classes)
        except ValueError as e:  # e.g. only one class in test data
            auc = np.nan
            sys.stderr.write(f"Warning: ROC-AUC cannot be computed: {e}\n")
        print(f"ROC-AUC: {auc}")
        print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
        print(f"MCC: {matthews_corrcoef(y_true, y_pred)}")
        print(f"Train F1 (mean over folds): {np.mean(train_f1s)}")
        if folds is not None:  # per-fold validation F1 is not meaningful for LOO
            print(f"Validation F1 (mean over folds): {np.mean(valid_f1s)}")
        f1 = f1_score(y_true, y_pred, average="macro")

        # confusion matrix with class names (rows: true class, columns: predicted class)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        results = [("cv_method", cv_method), ("n_estimators", model.get_params().get('n_estimators'))]
        if isinstance(model, xgb.XGBClassifier):
            results += [("class_weight", "balanced (sample_weight)")]
        elif 'class_weight' in model.get_params():
            results += [("class_weight", model.get_params()['class_weight'])]
        results += [("n_sequences", len(y_true))]
        results += [(f"confusion_matrix (true={t}, pred={p})", cm[i, j])
                    for i, t in enumerate(class_names) for j, p in enumerate(class_names)]
        results += [("ROC-AUC", auc), ("accuracy", accuracy_score(y_true, y_pred)),
                    ("MCC", matthews_corrcoef(y_true, y_pred)), ("F1 (macro)", f1),
                    ("train_F1 (macro, mean over folds)", np.mean(train_f1s))]
        if folds is not None:
            results.append(("validation_F1 (macro, mean over folds)", np.mean(valid_f1s)))
        write_cv_results(results)
        return f1, fold_ri


def write_cv_results(results, outfile="cv_results.tsv", append=False):
    # save the CV performance metrics as "metric<TAB>value"
    pd.DataFrame(results, columns=['metric', 'value'], dtype=object).to_csv(outfile, sep='\t', index=False,
                                                              mode='a' if append else 'w', header=not append)
    sys.stderr.write(f"{'Append' if append else 'Write'} CV results to {outfile}\n")


def fisher_mean_ci(r, level=0.95):
    # mean and 95% CI of correlations (one per fold) with Fisher's z transformation and t distribution
    z = np.arctanh(np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999))
    z = z[~np.isnan(z)]
    if len(z) < 2:
        return (np.tanh(z.mean()) if len(z) else np.nan), np.nan, np.nan
    half = stats.t.ppf((1 + level) / 2, len(z) - 1) * z.std(ddof=1) / np.sqrt(len(z))
    return np.tanh(z.mean()), np.tanh(z.mean() - half), np.tanh(z.mean() + half)


def write_ri_correlation(fold_ri, residue_importance, le=None):
    # Pearson's r and Spearman's rho between the RI of the full model and the RI of each CV fold model
    # (positions with RI = 0 in all folds and the full model are excluded)
    results = []
    for cls, group_ri in fold_ri.items():
        if cls not in residue_importance:
            continue
        full = pd.Series({pos: v['total'] for pos, v in residue_importance[cls].items()})
        table = pd.DataFrame(group_ri).reindex(full.index)
        keep = (table != 0).any(axis=1) | (full != 0)
        table, full = table[keep], full[keep]
        pearson = table.corrwith(full, method='pearson')
        spearman = table.corrwith(full, method='spearman')
        if len(fold_ri) == 1:
            suffix = ""
        else:
            cls_name = le.inverse_transform([cls])[0] if le is not None and cls != "maxshap" else cls
            suffix = f" [{cls_name}]"
        for name, r in [("Pearson_r", pearson), ("Spearman_rho", spearman)]:
            mean, low, high = fisher_mean_ci(r.values)
            results += [(f"RI_full_vs_fold_{name}_mean{suffix}", mean),
                        (f"RI_full_vs_fold_{name}_ci95_low{suffix}", low),
                        (f"RI_full_vs_fold_{name}_ci95_high{suffix}", high)]
        results.append((f"RI_full_vs_fold_n_positions{suffix}", int(keep.sum())))
    if results:  # nothing to write without per-fold RI (--cv-ri)
        write_cv_results(results, append=True)


def _shap_values(model, X, X_explain):
    explainer = shap.TreeExplainer(model, data=X, feature_perturbation="interventional")
    return explainer.shap_values(X_explain)


def compute_shap_values(model, X):
    # SHAP values of each sample are independent; split samples into chunks and
    # compute them in parallel with the number of CPUs of the model (n_jobs)
    n_jobs = model.get_params().get('n_jobs') or 1
    if n_jobs < 0:
        n_jobs = os.cpu_count()
    n_jobs = min(n_jobs, len(X))
    if n_jobs == 1:
        return _shap_values(model, X, X)
    chunks = np.array_split(np.arange(len(X)), n_jobs)
    values = Parallel(n_jobs=n_jobs)(delayed(_shap_values)(model, X, X.iloc[idx]) for idx in chunks)
    return np.concatenate(values, axis=0)


def per_residue_shapley(model, X, mode='classification', plot=True):
    fi_list = [(fe, float(im)) for fe, im in zip(X.columns, model.feature_importances_)]

    sys.stderr.write(f"Compute Shapley values for each features...\n")
    # Compute SHAP values
    shap_values = compute_shap_values(model, X)
    if plot:
        plot_shap_summary(shap_values, X, mode=mode)

    #print("# SHAP shape", np.array(shap_values).shape)
    if mode == 'classification' and shap_values.ndim == 3:
        mean_shap = np.mean(np.abs(shap_values), axis=0)
    else:
        mean_shap = np.mean(np.abs(shap_values), axis=0)

    #print("Mean SHAP shape:", mean_shap.shape)
    # df = pd.DataFrame(mean_shap)
    # df.to_csv("mean_shap.tsv", sep='\t')

    sc = mean_shap[0]
    n_class = 1
    if (isinstance(mean_shap[0], list) or isinstance(mean_shap[0], np.ndarray)) and len(mean_shap[0]) > 2:
        classes = model.classes_
        n_class = len(classes)
    else:
        classes = [0]

    #print("# classes = ", classes)
    shapley_residues = {} # (n_class, n_position, n_amino)
    for idx, cls in enumerate(classes):
        shapley_residues[cls] = {}
        for fi, sc in zip(fi_list, mean_shap):
            k, v = fi
            pos, amino = k.split('_')
            pos = int(pos.replace('p', ''))
            if pos not in shapley_residues[cls].keys():
                # initialize
                shapley_residues[cls][pos] = {aa:0 for aa in aminos}
                shapley_residues[cls][pos]['total'] = 0

            if isinstance(sc, list) or isinstance(sc, np.ndarray):
                shapley_residues[cls][pos]['total'] += float(sc[idx])
                shapley_residues[cls][pos][amino] = float(sc[idx])
            else:
                shapley_residues[cls][pos]['total'] += float(sc)
                shapley_residues[cls][pos][amino] = float(sc)

        # convert relative scores
        total_score = np.sum([s['total'] for pos, s in shapley_residues[cls].items()])
        for pos, sr in shapley_residues[cls].items():
            shapley_residues[cls][pos]['total'] /= total_score

    if n_class > 1:
        max_values = {}
        for idx, cls in enumerate(classes):
            for pos, v in shapley_residues[cls].items():
                if pos not in max_values.keys():
                    max_values[pos] = v
                if max_values[pos]['total'] < v['total']:
                    max_values[pos] = v
        shapley_residues["maxshap"] = max_values

    return shapley_residues


def per_residue_importance(model, X, mode='classification'):
    fi_list = [(fe, float(im)) for fe, im in zip(X.columns, model.feature_importances_)]

    # Important residues
    important_residues = {0:{}}
    
    for k, v in fi_list:
        msa_pos, amino = k.split('_')
        msa_pos = int(msa_pos.replace('p', ''))
        if msa_pos not in important_residues[0].keys():
            important_residues[0][msa_pos] = {aa:0 for aa in aminos}
            important_residues[0][msa_pos]['total'] = 0

        important_residues[0][msa_pos]['total'] += float(v)
        important_residues[0][msa_pos][amino] = float(v)

    return important_residues


def run_rfr(feature_table, seed=default_seed, mode='regression', explain="shap", folds=None,
            n_estimators=n_estimators_rf, cv_ri=False, n_jobs=None):
    sys.stderr.write(f"Run RFR\n")
    df = feature_table.drop(['seq_id', 'obj_param'], axis=1)
    y = feature_table['obj_param'].values.astype(float)
    X0 = pd.get_dummies(df)
    X = X0.values

    ids = feature_table['seq_id'].values
    X, y, ids = shuffle(X, y, ids, random_state=seed)

    sys.stderr.write("Compute Residue Importance...\n")
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)

    # Cross validation
    score, fold_ri = cross_valid(model, X, y, mode=mode, folds=folds, ids=ids,
                        feature_names=X0.columns, explain=explain, cv_ri=cv_ri)

    print(f"RMSE: {score}")

    # Extract important features (Compute Shapley values for each amino acid)
    model.fit(X, y)
    if explain == "shap":
        residue_importance = per_residue_shapley(model, X0, mode=mode)
    else:
        residue_importance = per_residue_importance(model, X0, mode=mode)
    write_ri_correlation(fold_ri, residue_importance)

    return residue_importance


def run_rfc(feature_table, seed=default_seed, mode='classification', explain="shap", folds=None,
            n_estimators=n_estimators_rf, cv_ri=False, n_jobs=None):
    sys.stderr.write(f"Run RFC \n")
    df = feature_table.drop(['seq_id', 'obj_param'], axis=1)
    y = feature_table['obj_param'].values
    X0 = pd.get_dummies(df)
    X = X0.values

    ids = feature_table['seq_id'].values
    X, y, ids = shuffle(X, y, ids, random_state=seed)

    sys.stderr.write("Compute Residue Importance...\n")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs, class_weight="balanced")

    # Cross validation
    score, fold_ri = cross_valid(model, X, y, mode=mode, folds=folds, ids=ids,
                        feature_names=X0.columns, explain=explain, cv_ri=cv_ri)

    print(f"F1-score: {score}")

    # Extract important features (Compute Shapley values for each amino acid)
    model.fit(X, y)
    if explain == "shap":
        residue_importance = per_residue_shapley(model, X0, mode=mode)
    else:
        residue_importance = per_residue_importance(model, X0, mode=mode)
    write_ri_correlation(fold_ri, residue_importance)

    return residue_importance


def run_xgbr(feature_table, seed=default_seed, mode='regression', explain="shap", folds=None,
             n_estimators=n_estimators_xgb, cv_ri=False, n_jobs=None):
    sys.stderr.write("Run XGBRegressor...\n")
    df = feature_table.drop(['seq_id', 'obj_param'], axis=1)
    y = feature_table['obj_param'].values.astype(float)
    X0 = pd.get_dummies(df)
    X = X0.values

    ids = feature_table['seq_id'].values
    X, y, ids = shuffle(X, y, ids, random_state=seed)

    sys.stderr.write("Compute Residue Importance...\n")
    model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)

    # Cross validation
    score, fold_ri = cross_valid(model, X, y, mode=mode, folds=folds, ids=ids,
                        feature_names=X0.columns, explain=explain, cv_ri=cv_ri)

    print(f"RMSE: {score}")

    # Extract important features (Compute Shapley values for each amino acid)
    model.fit(X, y)
    if explain == "shap":
        residue_importance = per_residue_shapley(model, X0, mode=mode)
    else:
        residue_importance = per_residue_importance(model, X0, mode=mode)
    write_ri_correlation(fold_ri, residue_importance)

    return residue_importance


def run_xgbc(feature_table, seed=default_seed, mode='classification', explain="shap", folds=None,
             n_estimators=n_estimators_xgb, cv_ri=False, n_jobs=None):
    sys.stderr.write("Run XGBClassifier...\n")
    df = feature_table.drop(['seq_id', 'obj_param'], axis=1)
    y0 = feature_table['obj_param'].values

    le = LabelEncoder()
    y = le.fit_transform(y0)

    X0 = pd.get_dummies(df)
    X = X0.values

    ids = feature_table['seq_id'].values
    X, y, ids = shuffle(X, y, ids, random_state=seed)

    sys.stderr.write("Compute Residue Importance...\n")
    model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)

    # Cross validation
    score, fold_ri = cross_valid(model, X, y, mode=mode, folds=folds, ids=ids,
                        feature_names=X0.columns, explain=explain, cv_ri=cv_ri, le=le)

    print(f"F1-score: {score}")

    # Extract important features (Compute Shapley values for each amino acid)
    fit_model(model, X, y)
    if explain == "shap":
        residue_importance = per_residue_shapley(model, X0, mode=mode)
    else:
        residue_importance = per_residue_importance(model, X0, mode=mode)
    write_ri_correlation(fold_ri, residue_importance, le=le)

    # multi-class: use class labels instead of the encoded classes (0, 1, ...) as keys,
    # so that output files are named by the labels (a single key 0 means "no class")
    if len(residue_importance) > 1:
        residue_importance = {(cls if cls == "maxshap" else le.inverse_transform([cls])[0]): ri
                              for cls, ri in residue_importance.items()}

    return residue_importance

