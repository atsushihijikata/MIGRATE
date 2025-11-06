import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from collections import Counter
import json
from scipy.stats import kruskal, spearmanr
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from .make_matrix import aminos


default_seed = 123
n_estimators = 100



def plot_confusion_matrix(y_true, y_pred):
    class_labels = list(set(y_true))

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    print("Confusion Matrix")
    print(f"{cm}")

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

    #print(mode)
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


def cross_valid(model, X, y, mode=None, seed=default_seed):
    # cross-validation

    if X.shape[0] < 60:  # for small datasets
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=10, shuffle=True, random_state=seed)

    y_true, y_pred = [], []

    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        if X_test.shape[0] == 1:
            y_pred.append(model.predict(X_test)[0])
            y_true.append(y_test[0])
        else:
            y_pred.extend(model.predict(X_test))
            y_true.extend(y_test)

    if mode == 'regression':
        plot_regression_scatter(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        rho, p_value = spearmanr(y_true, y_pred)
        print(f"corrcoef: {corr}")
        print(f"rho: {rho}, {p_value}")
        return np.sqrt(mse)

    elif mode == 'classification':
        plot_confusion_matrix(y_true, y_pred)
        return f1_score(y_true, y_pred, average="macro")


def per_residue_shapley(model, X, mode='classification'):
    fi_list = [(fe, float(im)) for fe, im in zip(X.columns, model.feature_importances_)]

    sys.stderr.write(f"Compute Shapley values for each features...\n")
    # Compute SHAP values
    explainer = shap.TreeExplainer(model, data=X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)
    plot_shap_summary(shap_values, X, mode=mode)

    #print("# SHAP shape", np.array(shap_values).shape)
    if mode == 'classification' and shap_values.ndim == 3:
        mean_shap = np.mean(np.abs(shap_values), axis=0)
    else:
        mean_shap = np.mean(np.abs(shap_values), axis=0)

    #print("Mean SHAP shape:", mean_shap.shape)
    df = pd.DataFrame(mean_shap)
    df.to_csv("mean_shap.tsv", sep='\t')

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

    print("shapley_residues=", shapley_residues.keys())

    return shapley_residues


def per_residue_importance(model, X, mode='classification'):
    fi_list = [(fe, float(im)) for fe, im in zip(X.columns, model.feature_importances_)]

    # Important residues
    important_residues = {0:{}}
    #print(f"fi_list: {len(fi_list)}")
    for k, v in fi_list:
        msa_pos, amino = k.split('_')
        msa_pos = int(msa_pos.replace('p', ''))
        if msa_pos not in important_residues.keys():
            important_residues[0][msa_pos] = {aa:0 for aa in aminos}
            important_residues[0][msa_pos]['total'] = 0

        important_residues[0][msa_pos]['total'] += float(v)
        important_residues[0][msa_pos][amino] = float(v)

    return important_residues


def run_rfr(feature_table, seed=default_seed, mode='regression', explain="shap"):
    sys.stderr.write(f"Run RandomForestRegressor for {feature_table}\n")
    df = feature_table.drop(['seq_id', 'obj_param'], axis=1)
    y = feature_table['obj_param'].values.astype(float)
    X0 = pd.get_dummies(df)
    X = X0.values

    X, y = shuffle(X, y, random_state=seed)

    sys.stderr.write("Compute Residue Importance...\n")
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)

    # Cross validation
    score = cross_valid(model, X, y, mode=mode)

    print(f"RMSE: {score}")

    # Extract important features (Compute Shapley values for each amino acid)
    model.fit(X, y)
    if explain == "shap":
        residue_importance = per_residue_shapley(model, X0, mode=mode)
    else:
        residue_importance = per_residue_importance(model, X0, mode=mode)

    return residue_importance


def run_rfc(feature_table, seed=default_seed, mode='classification', explain="shap"):
    sys.stderr.write(f"Run RFC \n")
    df = feature_table.drop(['seq_id', 'obj_param'], axis=1)
    y = feature_table['obj_param'].values
    X0 = pd.get_dummies(df)
    X = X0.values

    X, y = shuffle(X, y, random_state=seed)

    sys.stderr.write("Compute Residue Importance...\n")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)

    # Cross validation
    score = cross_valid(model, X, y, mode=mode)

    print(f"F1-score: {score}")

    # Extract important features (Compute Shapley values for each amino acid)
    model.fit(X, y)
    if explain == "shap":
        residue_importance = per_residue_shapley(model, X0, mode=mode)
    else:
        residue_importance = per_residue_importance(model, X0, mode=mode)

    return residue_importance


def run_xgbr(feature_table, seed=default_seed, mode='regression', explain="shap"):
    sys.stderr.write("Run XGBRegressor...\n")
    df = feature_table.drop(['seq_id', 'obj_param'], axis=1)
    y = feature_table['obj_param'].values.astype(float)
    X0 = pd.get_dummies(df)
    X = X0.values

    X, y = shuffle(X, y, random_state=seed)

    sys.stderr.write("Compute Residue Importance...\n")
    model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=seed)

    # Cross validation
    score = cross_valid(model, X, y, mode=mode)

    print(f"RMSE: {score}")

    # Extract important features (Compute Shapley values for each amino acid)
    model.fit(X, y)
    if explain == "shap":
        residue_importance = per_residue_shapley(model, X0, mode=mode)
    else:
        residue_importance = per_residue_importance(model, X0, mode=mode)

    return residue_importance


def run_xgbc(feature_table, seed=default_seed, mode='classification', explain="shap"):
    sys.stderr.write("Run XGBClassifier...\n")
    df = feature_table.drop(['seq_id', 'obj_param'], axis=1)
    y0 = feature_table['obj_param'].values

    le = LabelEncoder()
    y = le.fit_transform(y0)

    X0 = pd.get_dummies(df)
    X = X0.values

    X, y = shuffle(X, y, random_state=seed)

    print(len(list(set(y))))
    sys.stderr.write("Compute Residue Importance...\n")
    model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=seed)

    # Cross validation
    score = cross_valid(model, X, y, mode=mode)

    print(f"F1-score: {score}")

    # Extract important features (Compute Shapley values for each amino acid)
    model.fit(X, y)
    if explain == "shap":
        residue_importance = per_residue_shapley(model, X0, mode=mode)
    else:
        residue_importance = per_residue_importance(model, X0, mode=mode)

    return residue_importance

