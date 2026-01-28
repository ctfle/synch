import json
import os
import numpy as np
from itertools import chain

# Assuming same imports as original
from gates import t, sqrt_t
from synthesis import _substitute_duplicates
from utils import seq2mat, trace

try:
    import cupy as cp
    asnumpy = cp.asnumpy
except ModuleNotFoundError:
    cp = np
    asnumpy = np.asarray

nontrivial_cliffords = "sh"
trivial_cliffords = "xyz"
clifford_gates = nontrivial_cliffords + trivial_cliffords
nonclifford_gates = "tq"

# Original directory (input)
ORIG_DIR = (
    f"{os.path.dirname(os.path.abspath(__file__))}/assets/{nonclifford_gates}{clifford_gates}_short/"
)

# New directory for T-equivalent grouping (output)
NEW_DIR = (
    f"{os.path.dirname(os.path.abspath(__file__))}/assets/{nonclifford_gates}{clifford_gates}_tequiv_short/"
)

os.makedirs(NEW_DIR, exist_ok=True)

def count_t_equiv(seqstr):
    """Count T-equivalent gates: t=1, q=2, cliffords=0"""
    t_count = seqstr.count('t')
    q_count = seqstr.count('q')
    return t_count + 2 * q_count


def replace_and_drop(duplicates: dict[str, str], old: str, new: str) -> dict[str, str]:
    drop = []
    for i, (key, value) in enumerate(duplicates.items()):
        replaced_value = value.replace(old, new)
        duplicates[key] = replaced_value
        if key == replaced_value:
            drop.append(key)

    # drop the entries where key == value
    for key in drop:
        duplicates.pop(key)
    new_dupl = {}
    for key, value in duplicates.items():
        if len(value)<len(key):
            new_dupl[key] = value
        if len(value)>len(key):
            new_dupl[value] = key
        if len(value) == len(key):
            continue
         
    return new_dupl

def replace(sequences: list[str], old: str, new: str) -> list[str]:
    for i, seq in enumerate(sequences):
        sequences[i] = seq.replace(old, new)

    return sequences

if __name__ == "__main__":
    # Load base Clifford data (0 non-Cliffords, same for both)
    matrices_0 = np.load(f"{ORIG_DIR}tensor_0.npy")
    with open(f"{ORIG_DIR}sequences_0.json", "r", encoding="utf-8") as file:
        sequences_0 = json.load(file)
    with open(f"{ORIG_DIR}duplicates_0.json", "r", encoding="utf-8") as file:
        duplicates_0 = json.load(file)

    # Save base to new dir (unchanged)
    np.save(f"{NEW_DIR}tensor_0.npy", matrices_0)
    with open(f"{NEW_DIR}sequences_0.json", "w", encoding="utf-8") as file:
        json.dump(sequences_0, file, indent=4)
    with open(f"{NEW_DIR}duplicates_0.json", "w", encoding="utf-8") as file:
        json.dump(duplicates_0, file, indent=4)

    print("Base Clifford data copied (0 T-equivalent).")

    # Max T-equivalent to process (covers up to length=4 with q's)
    MAX_T_EQUIV = 8  # 4*q = 8 T-equivalent

    # Global structures for all T-equivalent levels - FIXED: use lists of matrices
    all_matrices = {0: matrices_0}
    all_sequences = {0: sequences_0}
    all_duplicates = {0: duplicates_0}

    # Load ALL original data first
    for orig_length in range(1, 5):
        matrices_k = np.load(f"{ORIG_DIR}tensor_{orig_length}.npy")
        with open(f"{ORIG_DIR}sequences_{orig_length}.json", "r") as f:
            sequences_k = json.load(f)
        with open(f"{ORIG_DIR}duplicates_{orig_length}.json", "r") as f:
            duplicates_k = json.load(f)

        duplicates_k = replace_and_drop(duplicates_k, "qq", "t")
        sequences_k = replace(sequences_k, "qq", "t")

        
        print(f"Loaded original length {orig_length}: {len(sequences_k)} sequences")

        # Classify each sequence by its T-equivalent count - FIXED: proper list append
        for idx, seqstr in enumerate(sequences_k):
            t_equiv = count_t_equiv(seqstr)
            if t_equiv > MAX_T_EQUIV:
                continue

            if t_equiv not in all_matrices:
                #all_matrices[t_equiv] = []
                all_sequences[t_equiv] = []
                all_duplicates[t_equiv] = {}

            # Extract matrix for this sequence (matrices_k is shape (2,N,2))
            matrix = matrices_k[:, idx, :]  # shape (2,2)

            #all_matrices[t_equiv].append(matrix)
            #print(all_matrices[t_equiv].shape)
            if all_matrices.get(t_equiv, None) is None:
                all_matrices[t_equiv] = matrix.reshape(2, 1, 2)
            else:
                all_matrices[t_equiv] = np.concatenate(
                    [all_matrices[t_equiv], matrix.reshape(2, 1, 2)], axis=1)

            all_sequences[t_equiv].append(seqstr)
            if dupl := duplicates_k.get(seqstr, None):
                all_duplicates[t_equiv][seqstr] = dupl

    # Now deduplicate WITHIN each T-equivalent bucket
    for t_equiv in np.arange(1, MAX_T_EQUIV + 0.5, 0.5):
        
        if t_equiv not in all_matrices: #or not all_matrices[t_equiv]:
            print(f"T-equivalent {t_equiv}: empty bucket")
            # Create empty files for consistency
            np.save(f"{NEW_DIR}tensor_{t_equiv}.npy", np.empty((2, 0, 2), dtype=complex))
            with open(f"{NEW_DIR}sequences_{t_equiv}.json", "w", encoding="utf-8") as f:
                json.dump([], f, indent=4)
            with open(f"{NEW_DIR}duplicates_{t_equiv}.json", "w", encoding="utf-8") as f:
                json.dump({}, f, indent=4)
            continue

        print(f"\nProcessing T-equivalent {t_equiv}: {len(all_sequences[t_equiv])} candidates")


        np.save(f"{NEW_DIR}tensor_{t_equiv}.npy", all_matrices[t_equiv])
        with open(f"{NEW_DIR}sequences_{t_equiv}.json", "w", encoding="utf-8") as f:
            json.dump(all_sequences[t_equiv], f, indent=4)
        with open(f"{NEW_DIR}duplicates_{t_equiv}.json", "w", encoding="utf-8") as f:
             json.dump(all_duplicates[t_equiv], f, indent=4)

        print(f"T-equivalent {t_equiv}: {len(all_sequences[t_equiv])} unique sequences saved")

    print(f"\nRegrouping complete! New files in {NEW_DIR}")
