import argparse
import time
import datetime
from .make_feature_table import make_feature_table
from .mmcif import get_mmcif
from .ml import run_rfc, run_rfr, run_xgbc, run_xgbr, read_cv_folds
from .map3d import map_importance
from .make_matrix import make_importance_matrix

def main():

    parser = argparse.ArgumentParser(description="MIGRATE: Machine learning-based Identification of Globally Adaptive Amino-acid Residues Associated with Tolerance to diverse Environments.")
    parser.add_argument("msa_file", help="MSA file")
    parser.add_argument("--target", help="Sequence Id in the MSA", required=True)
    parser.add_argument("--pdbid", help="PDB ID+Chain for mapping RI scores onto the 3D structure")
    parser.add_argument("--pdb-start", type=int, default=1, help="Start residue number of the sequence on the 3D structure.")
    parser.add_argument("--seed", type=int, default=123, help="Seed")
    parser.add_argument("--model", help="ML model. rf=RandomForest or xgb=XGBoost", default="rf")
    parser.add_argument("--n-estimators", type=int, default=None,
                        help="Number of trees. Default: 1000 for RandomForest (--model rf), 100 for XGBoost (--model xgb)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Number of CPUs for model training (-1: all CPUs). Default: 1 for RandomForest, all CPUs for XGBoost")
    parser.add_argument("--explain", choices=["shap", "importance"], default="shap",
                        help="Explain feature importance with SHAP values (shap) or the model's feature importance (importance: impurity-based for RF, gain for XGBoost). Default: shap")
    parser.add_argument("--mode", default='classification', help="Mode of ML. 'classification' or 'regression'.")
    parser.add_argument("--cv-ri", action="store_true",
                        help="Compute RI scores of the model of each CV fold (cv_ri_scores.tsv; time-consuming)")
    parser.add_argument("--cv-group", "--cv-fold", dest="cv_group", help="File of 'seq_id<TAB>group' (group: 1, 2, ... or train_only) for leave-one-group-out CV. Each group is used as a CV fold. Not class labels (labels are in the MSA header).")

    args = parser.parse_args()
    start_time = time.time()
    seed = args.seed
    target = args.target
    pdb_id = args.pdbid
    pdb_start = args.pdb_start
    mode = args.mode
    explain = args.explain
    model = args.model
    folds = read_cv_folds(args.cv_group) if args.cv_group else None
    # n_estimators / n_jobs: the default of each model (run_rf* / run_xgb*) is used unless given
    model_args = {'cv_ri': args.cv_ri}
    if args.n_estimators is not None:
        model_args['n_estimators'] = args.n_estimators
    if args.n_jobs is not None:
        model_args['n_jobs'] = args.n_jobs

    # Creating feature table from input MSA
    feature_table = make_feature_table(args.msa_file)

    residue_importance = {} # SHAP score for each residue

    # check target sequence existing
    if feature_table[feature_table['seq_id'].isin([target])].empty:
        print(f"Error: {target} does not exist in MSA.")
        exit()

    if mode == 'classification':
        if model == 'xgb':
            residue_importance = run_xgbc(feature_table, seed=seed, mode=mode, explain=explain, folds=folds,
                                          **model_args)
        else:
            residue_importance = run_rfc(feature_table, seed=seed, mode=mode, explain=explain, folds=folds,
                                         **model_args)

    elif mode == 'regression':
        if model == 'xgb':
            residue_importance = run_xgbr(feature_table, seed=seed, mode=mode, explain=explain, folds=folds,
                                          **model_args)
        else:
            residue_importance = run_rfr(feature_table, seed=seed, mode=mode, explain=explain, folds=folds,
                                         **model_args)

    for cls, ri_score in residue_importance.items():
        if cls == 0:
            cls = None
        map_importance(feature_table, ri_score, class_name=cls,
                       target_id=target, pdb_id=pdb_id, pdb_start=int(pdb_start))

        make_importance_matrix(feature_table, ri_score, class_name=cls, mode=mode, seed=seed)

    duration = time.time() - start_time
    print(f"Duration: {duration:.1f} s ({datetime.timedelta(seconds=round(duration))})")

