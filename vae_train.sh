#!/bin/bash

#SBATCH --time=hh:00:00   
#SBATCH --nodes=n         
#SBATCH --ntasks-per-node=a  
#SBATCH --mem=128G       
#SBATCH --gres=gpu:s     

python train_VAE.py \
    --embedding_file ../all_residue_embeddings.npy \
    --ids_file ../all_residue_ids.json \
    --protein_name prrsv \
    --VAE_checkpoint_location chkpts \
    --weights_file weights/sequence_weights.npy \
    --model_name_suffix esm \
    --model_parameters_location EVE/full-params.json \
    --training_logs_location logs \
    --seed 42