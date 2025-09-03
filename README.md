## This is the repository for FLEVO: Fitness Learning from Evolutionary Variants Over time
## Overview
We modified original training scripts from EVE, an unsupervised generative model of mutation effect from broader evolutionary sequences to handle ESM-2 protein language model embeddings to calculate fitness landscape in viral sequences of Porcine Reproductive and Respiratory Syndrome Viral Glycoprotein GP5.

## Usage
Computing FLEVO escape score is same as that of EVEscape.
1. Fitness: Measured as a log likelihood of a sequence with respect to the wild type. This is leart by the Bayesian VAE for input ESM-2 embeddings
2. Accessibility: WCN is calculated from Alphafold predicted 3D structure of PRRSV GP5
3. Dissimilarity: calculate difference in charge and hydrophobicity between the mutant residue and the wildtype 

The components are then standardized and fed into a temperature scaled logistic function, and we take the the log transform of the product of the 3 terms to obtain final escape scores. 

## Scripts
We have included python scripts to train the VAE, evaluate evolutionary index for all single point mutations in the training
 - [](scripts/process_protein_data.py) calculates the three EVEscape components 
 - [evescape_scores.py](scripts/evescape_scores.py) creates the final evescape scores and outputs scores and processed DMS data in [summaries_with_scores](./results/summaries_with_scores)
 
 The scripts folder also contains a python script [score_pandemic_strains.py](scripts/score_pandemic_strains.py) to calculate EVEscape scores for all strains in GISAID. The output strain scores (~150MB unzipped) can be downloaded as follows:
 ```
curl -o strain_scores_20230318.zip https://marks.hms.harvard.edu/evescape/strain_scores_20230318.zip
unzip strain_scores_20230318.zip
rm strain_scores_20230318.zip
```
The workflow of the scripts to create the data tables in [results](./results) needed for the main figures of the EVEscape paper is available in [evescape_summary.pdf](./evescape_summary.pdf). Additional data tables are available in the paper supplement. 

## Data requirements

## Generating EVE scores
We leverage the original [EVE codebase](https://github.com/OATML-Markslab/EVE) to compute the evolutionary indices used in EVEscape.

### Model training
The MSAs used to train the EVE models used in this project can be found in the supplemental material of the paper (Data S1). 

We modify the Bayesian VAE [training script](https://github.com/OATML-Markslab/EVE/blob/master/train_VAE.py) to support the following hyperparameter choices in the [MSA_processing](https://github.com/OATML-Markslab/EVE/blob/master/utils/data_utils.py) call:
- sequence re-weighting in MSA (theta): we choose a value of 0.01 that is better suited to viruses (Hopf et al., Riesselman et al.)
- fragment filtering (threshold_sequence_frac_gaps): we keep sequences in the MSA that align to at least 50% of the target sequence.
- position filtering (threshold_focus_cols_frac_gaps): we keep columns with at least 70% coverage, except for SARS-CoV-2 Spike for which we lower the required value to 30% in order to maximally cover experimental positions and significant pandemic sites.

We train 5 independent models with different random seeds.

### Model scoring
For the 5 independently-trained models, we compute [evolutionary indices](https://github.com/OATML-Markslab/EVE/blob/master/compute_evol_indices.py) sampling 20k times from the approximate posterior distribution (ie., num_samples_compute_evol_indices=20000). 

## Software requirements
The entire codebase is written in python. The corresponding environment may be created via conda and the provided [requirements.txt](./requirements.txt) file as follows:
```
conda config --add channels conda-forge
conda create --name evescape_env --file requirements.txt
conda activate evescape_env
```
The environment installs in minutes.

## Runtime
After collecting the training data, generating FLEVO escape scores for all single mutations runs in minutes.

## Reference


