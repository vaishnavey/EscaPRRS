## This is the repository for EscaPRRS: Escape scoring Evolutionary Variants Over time for Porcine Reproductive and Respiratory Syndrome Virus
## Overview
We modified original training scripts from EVE [training script](https://github.com/OATML-Markslab/EVE/blob/master/train_VAE.py), an unsupervised generative model of mutation effect from broader evolutionary sequences to handle ESM-2 protein language model embeddings to calculate fitness landscape in viral sequences of Porcine Reproductive and Respiratory Syndrome Viral Glycoprotein GP5.

## Usage
Computing EscaPRRS score is same as that of EVEscape.
1. Fitness: Measured as a log likelihood of a sequence with respect to the wild type. This is leart by the Bayesian VAE for input ESM-2 embeddings
2. Accessibility: WCN is calculated from Alphafold predicted 3D structure of PRRSV GP5
3. Dissimilarity: calculate difference in charge and hydrophobicity between the mutant residue and the wildtype 

The components are then standardized and fed into a temperature scaled logistic function, and we take the the log transform of the product of the 3 terms to obtain final escape scores. 

## Scripts
We have included python scripts to train the VAE, evaluate evolutionary index for all single point mutations in the training
We recommend referring the original EVE and EVEscape repositories to follow the steps conveniently.
A general workflow to calculate escape scores with EscaPRRS is given in [workflow_EscaPRRS.ipynb] (workflow_EscaPRRS.ipynb) with expected outputs.
  
## Data requirements
The following data files are required to run EscaPRRS
Training data: Protein sequences of same length sorted in ascending datetime order in FASTA format
Dissimilarity metric: Single mutation dissimilarity scores from EVEscape (https://github.com/OATML-Markslab/EVEscape/blob/main/data/aa_properties/dissimilarity_metrics.csv)
A 3D PDB structure file of the parent protein. (Predicted structures can also be used)
### Model training
A model fasta file (https://github.com/vaishnavey/EscaPRRS/blob/main/prrsv_ORF5_GP5.fasta) is provided to train the model.

## Software requirements
The entire codebase is written in python. The corresponding environment may be created via conda. Some dependencies and version used are listed in requirements.txt
```
conda config --add channels conda-forge
conda create --name escaprrs_env --file requirements.txt
conda activate escaprrs_env
```
Typical install time is 5 minutes

## Runtime
After collecting the training data, generating and visualizing FLEVO escape scores for all single mutations runs in minutes.




