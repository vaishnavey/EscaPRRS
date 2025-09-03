import numpy as np
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ESM2EmbeddingDataset:
    def __init__(self, embedding_file, ids_file, reference_sequence=None, weights_file=None, save_weights_to=None, theta=0.01):
        self.one_hot_encoding = np.load(embedding_file)  # shape (N, L, D)
        self.seq_len = self.one_hot_encoding.shape[1]
        self.embedding_dim = self.one_hot_encoding.shape[2]
        self.N = self.one_hot_encoding.shape[0]

        # Load sequence IDs
        if ids_file.endswith(".json"):
            with open(ids_file) as f:
                id_data = json.load(f)
                self.seq_ids = list(id_data.keys()) if isinstance(id_data, dict) else id_data
        else:
            with open(ids_file) as f:
                self.seq_ids = [line.strip() for line in f]

        # Set reference sequence
        if reference_sequence:
            self.reference_sequence = reference_sequence
        assert len(self.reference_sequence) == self.seq_len, "Reference sequence length must match embedding length"

        # Setup for EVE-style indexing
        self.focus_seq_trimmed = list(self.reference_sequence)
        self.focus_cols = list(range(self.seq_len))
        self.uniprot_focus_col_to_wt_aa_dict = {i + 1: aa for i, aa in enumerate(self.reference_sequence)}
        self.uniprot_focus_col_to_focus_idx = {i + 1: i for i in range(self.seq_len)}

        # Load or compute weights
        if weights_file and os.path.exists(weights_file):
            self.weights = np.load(weights_file)
            print(f"Loaded weights from {weights_file}")
        else:
            print("No weights file found. Computing using cosine similarity of average embeddings.")
            avg_emb = np.mean(self.one_hot_encoding, axis=1)  # shape (N, D)
            sim = cosine_similarity(avg_emb)
            self.weights = np.array([
                1.0 / np.sum(sim[i] > theta) if np.sum(sim[i] > theta) > 0 else 0.0
                for i in range(sim.shape[0])
            ])
            print(f"Computed sequence weights (theta = {theta})")

            if save_weights_to:
                os.makedirs(os.path.dirname(save_weights_to), exist_ok=True)
                np.save(save_weights_to, self.weights)
                print(f"Saved computed weights to: {save_weights_to}")

        self.Neff = float(np.sum(self.weights))
        self.num_sequences = self.N
        self.focus_start_loc = 0
        self.seq_name_to_sequence = {
            seq_id: emb for seq_id, emb in zip(self.seq_ids, self.one_hot_encoding)
        }

    def save_all_singles(self, output_filename):
        aas = "ACDEFGHIKLMNPQRSTVWY"
        all_mutations = []

        for i, wt in enumerate(self.reference_sequence):
            for mut in aas:
                if mut != wt:
                    mutation = f"{wt}{i+1}{mut}"
                    all_mutations.append(mutation)

        df = pd.DataFrame({'mutations': all_mutations})
        df.to_csv(output_filename, index=False)
        print(f"Saved all single mutations to {output_filename}")
