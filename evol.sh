#!/bin/bash

#SBATCH --time=hh:mi:ss   
#SBATCH --nodes=n        
#SBATCH --ntasks-per-node=n  
#SBATCH --mem=128G        
#SBATCH --gres=gpu:a     



python compute_evol_indices.py \
    --embedding_file ../all_mini_residue_embeddings.npy \
    --ids_file  ../all_avg_ids.json \
    --weights_file mini_weights/sequence_weights.npy \
    --protein_name protein123 \
    --VAE_checkpoint_location checkpoints \
    --model_name_suffix esm \
    --model_parameters_location EVE/default_model_params.json \
    --computation_mode all_singles \
    --all_singles_mutations_folder ./data/mutations \
    --output_evol_indices_location ./results/evol_indices \
    --num_samples_compute_evol_indices 100 \
    --batch_size 8
