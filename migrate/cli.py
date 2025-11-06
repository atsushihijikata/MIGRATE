import argparse
from .make_feature_table import make_feature_table
from .mmcif import get_mmcif
from .ml import run_rfc, run_rfr, run_xgbc, run_xgbr
from .map3d import map_importance
from .make_matrix import make_importance_matrix

def main():

    parser = argparse.ArgumentParser(description="MIGRATE: Machine learning-based Identification of Globally Adaptive Amino-acid Residues Associated with Tolerance to diverse Environments.")
    parser.add_argument("msa_file", help="MSA file")
    parser.add_argument("--target", help="Sequence Id in the MSA to assign RI scores", required=True)
    parser.add_argument("--pdbid", help="PDB ID+Chain to map the scores.")
    parser.add_argument("--pdb-start", help="Start residue number of the sequence with known 3D.")
    parser.add_argument("--seed", type=int, default=123, help="Seed")
    parser.add_argument("--model", help="ML model. rf=RandomForest or xgb=XGBoost", default="rf")
    parser.add_argument("--explain", help="Explain feature importance with SHAP or default feature importance (shap/default)", default="shap")
    parser.add_argument("--mode", default='classification', help="Mode of ML. 'classification' or 'regression'.")

    args = parser.parse_args()
    seed = args.seed
    target = args.target
    pdbid = args.pdbid
    start = args.pdb_start
    mode = args.mode
    explain = args.explain
    model = args.model

    # Creating feature table from input MSA
    feature_table = make_feature_table(args.msa_file)

    residue_importance = {} # SHAP score for each residue

    # check target sequence existing
    if feature_table[feature_table['seq_id'].isin([target])].empty:
        print(f"Error: {target} does not exist in MSA.")
        exit()

    if mode == 'classification':
        if model == 'xgb':
            residue_importance = run_xgbc(feature_table, seed=seed, mode=mode, explain=explain)
        else:
            residue_importance = run_rfc(feature_table, seed=seed, mode=mode, explain=explain)

    elif mode == 'regression':
        if model == 'xgb':
            residue_importance = run_xgbr(feature_table, seed=seed, mode=mode, explain=explain)
        else:
            residue_importance = run_rfr(feature_table, seed=seed, mode=mode, explain=explain)

    for cls, ri_score in residue_importance.items():
        if cls == 0:
            cls = None
        map_importance(feature_table, ri_score, class_name=cls,
                       target_id=target, pdb_id=pdbid, pdb_start=int(start))

        make_importance_matrix(feature_table, ri_score, class_name=cls)

