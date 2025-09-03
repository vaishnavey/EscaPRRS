import os
import numpy as np
import pandas as pd
import argparse
import pickle
import tqdm
from sklearn import mixture

# Placeholder for performance_helpers
class PerformanceHelpers:
    @staticmethod
    def compute_weighted_score_two_GMMs(X_pred, main_model, protein_model, cluster_index_main, cluster_index_protein, protein_weight):
        scores = protein_model.predict_proba(X_pred)[:, cluster_index_protein]
        return scores

    @staticmethod
    def compute_weighted_class_two_GMMs(X_pred, main_model, protein_model, cluster_index_main, cluster_index_protein, protein_weight):
        classes = protein_model.predict(X_pred)
        return classes

    @staticmethod
    def predictive_entropy_binary_classifier(scores):
        scores = np.clip(scores, 1e-10, 1 - 1e-10)
        entropy = - (scores * np.log2(scores) + (1 - scores) * np.log2(1 - scores))
        return entropy

    @staticmethod
    def compute_stats(scores):
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }

ph = PerformanceHelpers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GMM fit and EVE scores for a single protein')
    parser.add_argument('--input_evol_indices_location', type=str, required=True, help='Folder with evolutionary indices CSV')
    parser.add_argument('--input_evol_indices_filename_suffix', type=str, default='', help='File suffix for input CSV')
    parser.add_argument('--protein_list', type=str, required=True, help='File with protein name (single protein)')
    parser.add_argument('--output_eve_scores_location', type=str, required=True, help='Folder to save EVE scores')
    parser.add_argument('--output_eve_scores_filename_suffix', type=str, default='', help='Suffix for output files')
    parser.add_argument('--GMM_parameter_location', type=str, default='./gmm_models', help='Folder to save GMM models')
    parser.add_argument('--compute_EVE_scores', action='store_true', help='Compute EVE scores')
    parser.add_argument('--verbose', action='store_true', help='Print detailed info')
    args = parser.parse_args()

    # Read protein list (single protein)
    try:
        # Assume the file might have headers and multiple columns
        mapping_file = pd.read_csv(args.protein_list)
        if 'protein_name' not in mapping_file.columns:
            # If no headers, assume first column is protein_name
            mapping_file = pd.read_csv(args.protein_list, header=None, names=['protein_name'])
        mapping_file = mapping_file.dropna(subset=['protein_name'])  # Remove empty rows
        protein_list = np.unique(mapping_file['protein_name'].str.strip())  # Strip whitespace
        print("Loaded protein_list:", protein_list)
        if len(protein_list) != 1:
            raise ValueError(f"Expected exactly one protein in protein_list, found {len(protein_list)}: {protein_list}")
    except Exception as e:
        print(f"Error reading protein_list file: {e}")
        raise

    protein_name = protein_list[0]
    input_file = os.path.join(args.input_evol_indices_location, f"{protein_name}{args.input_evol_indices_filename_suffix}.csv")

    # Load evolutionary indices
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist")
    all_evol_indices = pd.read_csv(input_file, low_memory=False)[['protein_name', 'mutations', 'evol_indices']]
    all_evol_indices = all_evol_indices.drop_duplicates()
    X_train = np.array(all_evol_indices['evol_indices']).reshape(-1, 1)

    if args.verbose:
        print(f"Loaded {len(X_train)} evolutionary indices for protein {protein_name}")

    # Train GMM
    dict_models = {}
    dict_pathogenic_cluster_index = {}
    os.makedirs(args.GMM_parameter_location, exist_ok=True)

    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full', max_iter=1000, n_init=30, tol=1e-4)
    gmm.fit(X_train)

    dict_models[protein_name] = gmm
    dict_models['main'] = gmm
    pathogenic_cluster_index = np.argmax(np.array(gmm.means_).flatten())
    dict_pathogenic_cluster_index[protein_name] = pathogenic_cluster_index
    dict_pathogenic_cluster_index['main'] = pathogenic_cluster_index

    if args.verbose:
        print(f"Pathogenic cluster index: {pathogenic_cluster_index}")
        print(f"GMM weights: {gmm.weights_}")
        print(f"GMM means: {gmm.means_}")
        print(f"GMM covariances: {gmm.covariances_}")

    os.makedirs(os.path.join(args.GMM_parameter_location, args.output_eve_scores_filename_suffix), exist_ok=True)
    pickle.dump(dict_models, open(os.path.join(args.GMM_parameter_location, args.output_eve_scores_filename_suffix, f'GMM_model_dictionary_{args.output_eve_scores_filename_suffix}'), 'wb'))
    pickle.dump(dict_pathogenic_cluster_index, open(os.path.join(args.GMM_parameter_location, args.output_eve_scores_filename_suffix, f'GMM_pathogenic_cluster_index_dictionary_{args.output_eve_scores_filename_suffix}'), 'wb'))

    if args.compute_EVE_scores:
        all_scores = all_evol_indices.copy()
        all_scores['EVE_scores'] = np.nan
        all_scores['EVE_classes_100_pct_retained'] = ""

        X_test = np.array(all_scores['evol_indices']).reshape(-1, 1)
        mutation_scores = ph.compute_weighted_score_two_GMMs(
            X_pred=X_test,
            main_model=dict_models['main'],
            protein_model=dict_models[protein_name],
            cluster_index_main=dict_pathogenic_cluster_index['main'],
            cluster_index_protein=dict_pathogenic_cluster_index[protein_name],
            protein_weight=0.0
        )
        gmm_class = ph.compute_weighted_class_two_GMMs(
            X_pred=X_test,
            main_model=dict_models['main'],
            protein_model=dict_models[protein_name],
            cluster_index_main=dict_pathogenic_cluster_index['main'],
            cluster_index_protein=dict_pathogenic_cluster_index[protein_name],
            protein_weight=0.0
        )
        gmm_class_label = pd.Series(gmm_class).map(lambda x: 'Pathogenic' if x == pathogenic_cluster_index else 'Benign')

        all_scores['EVE_scores'] = mutation_scores
        all_scores['EVE_classes_100_pct_retained'] = gmm_class_label

        all_scores['uncertainty'] = ph.predictive_entropy_binary_classifier(all_scores['EVE_scores'])

        os.makedirs(args.output_eve_scores_location, exist_ok=True)
        output_file = os.path.join(args.output_eve_scores_location, f'EVE_scores_{protein_name}_{args.output_eve_scores_filename_suffix}.csv')
        all_scores.to_csv(output_file, index=False)

        if args.verbose:
            print(f"Saved EVE scores to {output_file}")
            print("Score stats:", ph.compute_stats(all_scores['EVE_scores']))