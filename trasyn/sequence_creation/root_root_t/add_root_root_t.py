import json
import os
import numpy as np
from sympy import sequence

from trasyn.utils import replace, replace_and_drop, count_t_equiv, seq2mat

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
MAX_COST = 8  # max value = 6*q = 15 T-equivalent.


# Original directory (input)
ORIG_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/merged/{nonclifford_gates}{clifford_gates}_tequiv_large_cost_2.5_all/"

# New directory for T-equivalent grouping (output)
NEW_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/merged/root_root_t/{nonclifford_gates}r{clifford_gates}_tequiv_large_q_2.5_r_4/"

os.makedirs(NEW_DIR, exist_ok=True)


def variants_with_one_r(seq: str):
    result = []
    for i, ch in enumerate(seq):
        if ch in ("t", "q"):
            result.append(seq[:i] + "r" + seq[i + 1 :])
    return result


def eval_cost(seqstr):
    """Count T-equivalent gates: t=1, q=2, cliffords=0"""
    t_count = seqstr.count("t")
    q_count = seqstr.count("q")
    r_count = seqstr.count("r")
    return t_count + 2.5 * q_count + 4 * r_count


if __name__ == "__main__":
    sequences = {}
    duplicates = {}
    tensors = {}

    new_sequences = {cost: [] for cost in np.arange(0, MAX_COST + 0.5, 0.5)}
    new_duplicates = {}
    new_tensors = {}

    for cost in np.arange(1, MAX_COST + 0.5, 0.5):
        # load corresponding sequences, duplicates and tensors
        mat = np.load(f"{ORIG_DIR}tensor_{cost}.npy")
        with open(f"{ORIG_DIR}sequences_{cost}.json", "r", encoding="utf-8") as file:
            seq = json.load(file)
        with open(f"{ORIG_DIR}duplicates_{cost}.json", "r", encoding="utf-8") as file:
            dup = json.load(file)

        sequences[cost] = seq
        duplicates[cost] = dup
        tensors[cost] = mat

        for s in seq:
            variants = variants_with_one_r(s)
            for new_s in variants:
                if (new_cost := eval_cost(new_s)) <= MAX_COST:
                    new_sequences[new_cost].append(new_s)

                    matrix = seq2mat(new_s)
                    if new_tensors.get(new_cost, None) is None:
                        new_tensors[new_cost] = matrix.reshape(2, 1, 2)
                    else:
                        new_tensors[new_cost] = np.concatenate(
                            [new_tensors[new_cost], matrix.reshape(2, 1, 2)], axis=1
                        )

    for cost, seqs in sequences.items():
        seqs.extend(new_sequences[cost])

    for cost, mat in tensors.items():
        if new_tensors.get(cost, None) is not None:
            tensors[cost] = np.concatenate([mat, new_tensors[cost]], axis=1)

    # save the extended sequences:
    for cost in np.arange(1, MAX_COST + 0.5, 0.5):
        np.save(f"{NEW_DIR}tensor_{cost}.npy", tensors[cost])
        with open(f"{NEW_DIR}sequences_{cost}.json", "w", encoding="utf-8") as file:
            json.dump(sequences[cost], file, indent=4)
        with open(f"{NEW_DIR}duplicates_{cost}.json", "w", encoding="utf-8") as file:
            json.dump(duplicates[cost], file, indent=4)
