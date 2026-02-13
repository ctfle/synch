import json
import os
import numpy as np
from tqdm import tqdm
from trasyn.utils import seq2mat, count_t_equiv

try:
    import cupy as cp

    asnumpy = cp.asnumpy
except ModuleNotFoundError:
    cp = np
    asnumpy = np.asarray

# This script merges sequence. It loads to sequence to merge into (S0) and the sequence file to
# merge from (S1)
# for each sequence in S1 checks if is contained in S0. If not adds sequence to S0 and concatenates
# corresponding matrix.


nontrivial_cliffords = "sh"
trivial_cliffords = "xyz"
clifford_gates = nontrivial_cliffords + trivial_cliffords
nonclifford_gates = "tq"
nonclifford_gates_merge_from = "t"
# Original maximum sequence length
MAX_LEN = 6
# Max T-equivalent to process (covers up to length=4 with q's)
MAX_LEN = 10  # max value = 6*q = 15 T-equivalent.

# Original directory (input)
ORIG_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/{nonclifford_gates}{clifford_gates}_tequiv_large_cost_2.5/"
# Dir to merge from
MERGE_FROM_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/{nonclifford_gates_merge_from}{clifford_gates}_large/"
MERGE_FROM_SEQ = MERGE_FROM_DIR + "sequences_11.json"
MERGE_FROM_DUP = MERGE_FROM_DIR + "duplicates_11.json"
# New directory for T-equivalent grouping (output)
NEW_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/merged/{nonclifford_gates}{clifford_gates}_tequiv_large_cost_2.5_all/"
os.makedirs(NEW_DIR, exist_ok=True)


if __name__ == "__main__":

    # load sequences and duplicates to merge from
    with open(MERGE_FROM_SEQ, "r", encoding="utf-8") as file:
        merge_from_sequence = json.load(file)

    with open(MERGE_FROM_DUP, "r", encoding="utf-8") as file:
        merge_from_duplicates = json.load(file)

    # load the corresponding files
    for cost in tqdm(np.arange(10, MAX_LEN +0.5, 0.5)):
        with open(f"{ORIG_DIR}sequences_{cost}.json", "r", encoding="utf-8") as file:
            sequences = json.load(file)
        with open(f"{ORIG_DIR}duplicates_{cost}.json", "r", encoding="utf-8") as file:
            duplicates = json.load(file)
        tensor = np.load(f"{ORIG_DIR}tensor_{cost}.npy")

        #extend the sequences and matrices
        for seq in merge_from_sequence:
            if count_t_equiv(seq) == cost:
                if seq not in sequences:
                    sequences.append(seq)
                    matrix = seq2mat(seq).reshape(2,1,2)
                    tensor = np.concatenate((tensor, matrix), axis=1)
                else:
                    continue

        # extend the duplicates
        for dup in merge_from_duplicates:
            if count_t_equiv(dup) == cost:
                if dup not in duplicates:
                    duplicates[dup] = merge_from_duplicates[dup]
                else:
                    continue

        # save under new dir
        np.save(f"{NEW_DIR}tensor_{cost}.npy", tensor)
        with open(f"{NEW_DIR}sequences_{cost}.json", "w", encoding="utf-8") as file:
            json.dump(sequences, file, indent=4)
        with open(f"{NEW_DIR}duplicates_{cost}.json", "w", encoding="utf-8") as file:
            json.dump(duplicates, file, indent=4)
