import sys
import pandas as pd
from make_feature_table import make_feature_table
from make_matrix import aminos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import shap


def test_shap(ft):

    df = feature_table.drop(['seq_id', 'obj_param'], axis=1)
    y = feature_table['obj_param'].values
    X0 = pd.get_dummies(df)
    X = X0.values
    n_estimators = 100
    seed = 123

    sys.stderr.write("Compute Residue Importance...\n")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    model.fit(X, y)

    fi_list = [(fe, float(im)) for fe, im in zip(X0.columns, model.feature_importances_)]
    print(len(fi_list))

    # Compute SHAP values
    explainer = shap.TreeExplainer(model, data=X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)

    print(type(X0))
    print(shap_values.shape)
    print(X0.shape)
    shap.summary_plot(shap_values[:, :, 1], X0)
    #shap.summary_plot(shap_values[1], X, plot_type="bar", show=True)
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    #shap.summary_plot(mean_shap)
    print(mean_shap.shape)
    #for m in mean_shap[:, 0]:
        #print(m)




if __name__ == '__main__':
    msa_file = sys.argv[1]
    feature_table = make_feature_table(msa_file)

    test_shap(feature_table)
    exit()

    positions = list(feature_table.columns[2:])
    #print(positions)
    position_residue_matrix = {pos: {a: 0 for a in aminos} for pos in positions}

    for pos in positions:
        print(pos)
        for res in feature_table[pos]:
            #print(row)
            position_residue_matrix[pos][res] += 1

        break
        #for i, pos in enumerate(positions):
            #res = row[i+2]
            #print(row, pos)
            #position_residue_matrix[pos][res] += 1

