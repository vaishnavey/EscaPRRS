import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch

from EVE import VAE_model
from utils import data_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute evolutionary indices using ESM2 embeddings')
    parser.add_argument('--embedding_file', type=str, required=True)
    parser.add_argument('--ids_file', type=str, required=True)
    parser.add_argument('--weights_file', type=str)
    parser.add_argument('--protein_name', type=str, required=True)
    parser.add_argument('--VAE_checkpoint_location', type=str, required=True)
    parser.add_argument('--model_name_suffix', type=str, default='esm')
    parser.add_argument('--model_parameters_location', type=str, required=True)
    parser.add_argument('--computation_mode', type=str, choices=['all_singles', 'input_mutations_list'], required=True)
    parser.add_argument('--all_singles_mutations_folder', type=str)
    parser.add_argument('--mutations_location', type=str)
    parser.add_argument('--output_evol_indices_location', type=str, required=True)
    parser.add_argument('--output_evol_indices_filename_suffix', default='', type=str)
    parser.add_argument('--num_samples_compute_evol_indices', type=int, required=True)
    parser.add_argument('--batch_size', default=256, type=int)
    args = parser.parse_args()

    # Define reference sequence
    reference_sequence = "MLEKCLTAGYCSQLLFFWCIVPFCFAALVNAASNSSSHLQLIYNLTICELNGTDWLNQKFDWAVETFVIFPVLTHIVSYGALTTSHFLDTAGLITVSTAGYYHGRYVLSSIYAVFALAALICFVIRLTKNCMSWRYSCTRYTNFLLDTKGNLYRWRSPVVIERRGKVEVGDHLIDLKRVVLDGSAATPITKISAEQWGRP"

    # Load dataset
    data = data_utils.ESM2EmbeddingDataset(
        embedding_file=args.embedding_file,
        ids_file=args.ids_file,
        reference_sequence=reference_sequence,
        weights_file=args.weights_file,
        save_weights_to=None,
        theta=0.01
    )

    # Mutation file generation
    if args.computation_mode == "all_singles":
        assert args.all_singles_mutations_folder, "Missing --all_singles_mutations_folder"
        os.makedirs(args.all_singles_mutations_folder, exist_ok=True)
        mut_file = os.path.join(args.all_singles_mutations_folder, f"{args.protein_name}_all_singles.csv")
        data.save_all_singles(mut_file)
        args.mutations_location = mut_file
    else:
        assert args.mutations_location, "Missing --mutations_location"
        args.mutations_location = os.path.join(args.mutations_location, f"{args.protein_name}.csv")

    # Load model
    model_name = f"{args.protein_name}_{args.model_name_suffix}"
    with open(args.model_parameters_location) as f:
        model_params = json.load(f)

    model = VAE_model.VAE_model(
        model_name=model_name,
        data=data,
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        random_seed=42
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint_path = os.path.join(args.VAE_checkpoint_location, f"{model_name}_final")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from: {checkpoint_path}")
    except FileNotFoundError:
        print(f" Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Compute evolutionary indices
    print("Computing evolutionary indices...")
    valid_mutations, evol_indices, _, _ = model.compute_evol_indices(
        msa_data=data,
        list_mutations_location=args.mutations_location,
        num_samples=args.num_samples_compute_evol_indices,
        batch_size=args.batch_size
    )

    # Handle mismatches safely
    if len(valid_mutations) != len(evol_indices):
        print(f" Mismatch: {len(valid_mutations)} mutations vs {len(evol_indices)} indices. Truncating to minimum length.")
        min_len = min(len(valid_mutations), len(evol_indices))
        valid_mutations = valid_mutations[:min_len]
        evol_indices = evol_indices[:min_len]

    # Convert to flat scalar list
    evol_indices_flat = [
        float(np.mean(elbo)) if hasattr(elbo, '__len__') else float(elbo)
        for elbo in evol_indices
    ]

    # Save output
    os.makedirs(args.output_evol_indices_location, exist_ok=True)
    output_file = os.path.join(
        args.output_evol_indices_location,
        f"{args.protein_name}_{args.num_samples_compute_evol_indices}_samples{args.output_evol_indices_filename_suffix}.csv"
    )

    df = pd.DataFrame({
        'protein_name': [args.protein_name] * len(valid_mutations),
        'mutations': valid_mutations,
        'evol_indices': evol_indices_flat
    })
    df.to_csv(output_file, index=False)
    print(f"Saved evolutionary indices to: {output_file}")
