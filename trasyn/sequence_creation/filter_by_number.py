import json
import os
import numpy as np

from trasyn.utils import replace, replace_and_drop, count_t_equiv

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

# Original maximum sequence length
MAX_LEN = 6

# Max T-equivalent to process (covers up to length=4 with q's)
MAX_T_EQUIV = 7.5  # max value = 6*q = 15 T-equivalent.

MAX_NUM_SQRTT = 1

# Original directory (input)
ORIG_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/{nonclifford_gates}{clifford_gates}_large/"

# New directory for T-equivalent grouping (output)
NEW_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/filtered/{nonclifford_gates}{clifford_gates}_tequiv_large_cost_2.5_max_num_sqrtt_{MAX_NUM_SQRTT}/"

os.makedirs(NEW_DIR, exist_ok=True)


if __name__ == "__main__":
    # Load base Clifford data (0 non-Cliffords, same for both)
    matrices_0 = np.load(f"{ORIG_DIR}tensor_0.npy")
    with open(f"{ORIG_DIR}sequences_0.json", "r", encoding="utf-8") as file:
        sequences_0 = json.load(file)
    with open(f"{ORIG_DIR}duplicates_0.json", "r", encoding="utf-8") as file:
        duplicates_0 = json.load(file)

    # Save base to new dir (unchanged)
    np.save(f"{NEW_DIR}tensor_0.0.npy", matrices_0)
    with open(f"{NEW_DIR}sequences_0.0.json", "w", encoding="utf-8") as file:
        json.dump(sequences_0, file, indent=4)
    with open(f"{NEW_DIR}duplicates_0.0.json", "w", encoding="utf-8") as file:
        json.dump(duplicates_0, file, indent=4)

    print("Base Clifford data copied (0 T-equivalent).")

    # Global structures for all T-equivalent levels - FIXED: use lists of matrices
    all_matrices = {0: matrices_0}
    all_sequences = {0: sequences_0}
    all_duplicates = {0: duplicates_0}

    # Load ALL original data first
    for orig_length in range(1, MAX_LEN + 1):
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
            if seqstr.count("q") <= MAX_NUM_SQRTT:
                t_equiv = count_t_equiv(seqstr)
                if t_equiv == 7:
                    print("7")
                if t_equiv > MAX_T_EQUIV:
                    continue

                if t_equiv not in all_matrices:
                    # all_matrices[t_equiv] = []
                    all_sequences[t_equiv] = []
                    all_duplicates[t_equiv] = {}

                # Extract matrix for this sequence (matrices_k is shape (2,N,2))
                matrix = matrices_k[:, idx, :]  # shape (2,2)

                # all_matrices[t_equiv].append(matrix)
                # print(all_matrices[t_equiv].shape)
                if all_matrices.get(t_equiv, None) is None:
                    all_matrices[t_equiv] = matrix.reshape(2, 1, 2)
                else:
                    all_matrices[t_equiv] = np.concatenate(
                        [all_matrices[t_equiv], matrix.reshape(2, 1, 2)], axis=1
                    )

                all_sequences[t_equiv].append(seqstr)
                if dupl := duplicates_k.get(seqstr, None):
                    all_duplicates[t_equiv][seqstr] = dupl

    # Now deduplicate WITHIN each T-equivalent bucket
    for t_equiv in np.arange(1, MAX_T_EQUIV + 0.5, 0.5):
        if t_equiv not in all_matrices:  # or not all_matrices[t_equiv]:
            print(f"T-equivalent {t_equiv}: empty bucket")
            # Create empty files for consistency
            np.save(
                f"{NEW_DIR}tensor_{t_equiv}.npy", np.empty((2, 0, 2), dtype=complex)
            )
            with open(f"{NEW_DIR}sequences_{t_equiv}.json", "w", encoding="utf-8") as f:
                json.dump([], f, indent=4)
            with open(
                f"{NEW_DIR}duplicates_{t_equiv}.json", "w", encoding="utf-8"
            ) as f:
                json.dump({}, f, indent=4)
            continue

        print(
            f"\nProcessing T-equivalent {t_equiv}: {len(all_sequences[t_equiv])} candidates"
        )

        np.save(f"{NEW_DIR}tensor_{t_equiv}.npy", all_matrices[t_equiv])
        with open(f"{NEW_DIR}sequences_{t_equiv}.json", "w", encoding="utf-8") as f:
            json.dump(all_sequences[t_equiv], f, indent=4)
        with open(f"{NEW_DIR}duplicates_{t_equiv}.json", "w", encoding="utf-8") as f:
            json.dump(all_duplicates[t_equiv], f, indent=4)

        print(
            f"T-equivalent {t_equiv}: {len(all_sequences[t_equiv])} unique sequences saved"
        )

    print(f"\nRegrouping complete! New files in {NEW_DIR}")
