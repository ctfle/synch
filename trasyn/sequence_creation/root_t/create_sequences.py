import json
import os
import shutil
from itertools import product, chain
from pathlib import Path

import numpy as np

from trasyn.gates import sqrt_t, t
from trasyn.synthesis import _substitute_duplicates
from trasyn.utils import replace, replace_and_drop, count_t_equiv, seq2mat, trace

try:
    import cupy as cp

    asnumpy = cp.asnumpy
except ModuleNotFoundError:
    cp = np
    asnumpy = np.asarray


class SequenceCreator:
    def __init__(
        self,
        trivial_clifford_gates: str = "xyz",
        nontrivial_clifford_gates: str = "sh",
        non_clifford_gates: str = "tq",
        max_t_equiv: int = 8,
        max_len: int = 6
    ):
        """
        The strategy we use to create all sequences that contain sqrtT and T gates (annotated t
        and q here) works as follows:
        1. We create all sequences with an arbitrary number of t or q gates up to given number
        (max_len). We do this up to max_len 6

        2. We loop through the sequences and regroup them according to cost

        3. To ensure we create all the sequences (including those which don't contain any q gates)
        we merge our generated sequences with the ones that only have t gates as non-Clifford gates.
        Note that for max_len=6 and a cost of 2.5 for q gates we can create all sequences up to
        a total cost of 8.

        Note: There might be simpler approaches that achieve the same end result. This approach
        was build on top of the approach provided by the original trasyn repo.
        
        """

        self.trivial_clifford_gates = trivial_clifford_gates
        self.nontrivial_clifford_gates = nontrivial_clifford_gates
        self.clifford_gates = (
            self.nontrivial_clifford_gates + self.trivial_clifford_gates
        )
        self.non_clifford_gates = non_clifford_gates
        self.asset_dir = f"{os.path.dirname(os.path.abspath(__file__))}/../../assets/{self.non_clifford_gates}{self.clifford_gates}/"
        Path(self.asset_dir).mkdir(parents=True, exist_ok=True)
        self._t_asset_dir = f"{os.path.dirname(os.path.abspath(__file__))}/../../assets/t{self.clifford_gates}/"
        self.max_len = max_len
        self.max_t_equiv = max_t_equiv
        self._temp_dir = f"{os.path.dirname(os.path.abspath(__file__))}/../../assets/temp/{self.non_clifford_gates}{self.clifford_gates}/"

    def generate_unique_sequences(
        self,
    ):
        # create the all sequences with either t or q up to self.
        Path(self._temp_dir).mkdir(parents=True, exist_ok=True)
        self._create()
        self._regroup_with_cost()
        self._merge_sequences()
        shutil.rmtree(self._temp_dir)

    def _regroup_with_cost(self):
        """Loads the bare sequences and regroups them according to cost."""
        matrices_0 = np.load(f"{self._temp_dir}tensor_0.npy")
        with open(f"{self._temp_dir}sequences_0.json", "r", encoding="utf-8") as file:
            sequences_0 = json.load(file)
        with open(f"{self._temp_dir}duplicates_0.json", "r", encoding="utf-8") as file:
            duplicates_0 = json.load(file)

        # Save base to new dir (unchanged)
        np.save(f"{self.asset_dir}tensor_0.0.npy", matrices_0)
        with open(f"{self.asset_dir}sequences_0.0.json", "w", encoding="utf-8") as file:
            json.dump(sequences_0, file, indent=4)
        with open(
            f"{self.asset_dir}duplicates_0.0.json", "w", encoding="utf-8"
        ) as file:
            json.dump(duplicates_0, file, indent=4)

        print("Base Clifford data copied (0 T-equivalent).")

        # Global structures for all T-equivalent levels - FIXED: use lists of matrices
        all_matrices = {0: matrices_0}
        all_sequences = {0: sequences_0}
        all_duplicates = {0: duplicates_0}

        # Load all original data first
        for orig_length in range(1, self.max_len + 1):
            matrices_k = np.load(f"{self._temp_dir}tensor_{orig_length}.npy")
            with open(f"{self._temp_dir}sequences_{orig_length}.json", "r") as f:
                sequences_k = json.load(f)
            with open(f"{self._temp_dir}duplicates_{orig_length}.json", "r") as f:
                duplicates_k = json.load(f)

            duplicates_k = replace_and_drop(duplicates_k, "qq", "t")
            sequences_k = replace(sequences_k, "qq", "t")

            print(f"Loaded original length {orig_length}: {len(sequences_k)} sequences")

            # Classify each sequence by its T-equivalent count - FIXED: proper list append
            for idx, seqstr in enumerate(sequences_k):
                t_equiv = count_t_equiv(seqstr)
                if t_equiv > self.max_t_equiv:
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

        # Now deduplicate within each T-equivalent bucket
        for t_equiv in np.arange(0.0, self.max_t_equiv + 0.5, 0.5):
            if t_equiv not in all_matrices:  # or not all_matrices[t_equiv]:
                print(f"T-equivalent {t_equiv}: empty bucket")
                # Create empty files for consistency
                np.save(
                    f"{self.asset_dir}tensor_{t_equiv}.npy",
                    np.empty((2, 0, 2), dtype=complex),
                )
                with open(
                    f"{self.asset_dir}sequences_{t_equiv}.json", "w", encoding="utf-8"
                ) as f:
                    json.dump([], f, indent=4)
                with open(
                    f"{self.asset_dir}duplicates_{t_equiv}.json", "w", encoding="utf-8"
                ) as f:
                    json.dump({}, f, indent=4)
                continue

            print(
                f"\nProcessing T-equivalent {t_equiv}: {len(all_sequences[t_equiv])} candidates"
            )

            np.save(f"{self.asset_dir}tensor_{t_equiv}.npy", all_matrices[t_equiv])
            with open(
                f"{self.asset_dir}sequences_{t_equiv}.json", "w", encoding="utf-8"
            ) as f:
                json.dump(all_sequences[t_equiv], f, indent=4)
            with open(
                f"{self.asset_dir}duplicates_{t_equiv}.json", "w", encoding="utf-8"
            ) as f:
                json.dump(all_duplicates[t_equiv], f, indent=4)

            print(
                f"T-equivalent {t_equiv}: {len(all_sequences[t_equiv])} unique sequences saved"
            )
        print(f"\nRegrouping complete! New files in {self.asset_dir}")

    def _merge_sequences(self):
        """To get all sequenes we need to include the ones with only T gates"""
        merge_from_seq = self._t_asset_dir + "sequences_11.json"
        merge_from_dup = self._t_asset_dir + "duplicates_11.json"
        # load sequences and duplicates to merge from
        with open(merge_from_seq, "r", encoding="utf-8") as file:
            merge_from_sequence = json.load(file)

        with open(merge_from_dup, "r", encoding="utf-8") as file:
            merge_from_duplicates = json.load(file)

        # load the corresponding files
        for cost in np.arange(0.0, self.max_t_equiv + 0.5, 0.5):
            with open(
                f"{self.asset_dir}sequences_{cost}.json", "r", encoding="utf-8"
            ) as file:
                sequences = json.load(file)
            with open(
                f"{self.asset_dir}duplicates_{cost}.json", "r", encoding="utf-8"
            ) as file:
                duplicates = json.load(file)
            tensor = np.load(f"{self.asset_dir}tensor_{cost}.npy")

            # extend the sequences and matrices
            for seq in merge_from_sequence:
                if count_t_equiv(seq) == cost:
                    if seq not in sequences:
                        sequences.append(seq)
                        matrix = seq2mat(seq).reshape(2, 1, 2)
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
            np.save(f"{self.asset_dir}tensor_{cost}.npy", tensor)
            with open(
                f"{self.asset_dir}sequences_{cost}.json", "w", encoding="utf-8"
            ) as file:
                json.dump(sequences, file, indent=4)
            with open(
                f"{self.asset_dir}duplicates_{cost}.json", "w", encoding="utf-8"
            ) as file:
                json.dump(duplicates, file, indent=4)

    def _create(self):
        try:
            matrices = np.load(f"{self._temp_dir}tensor_0.npy")
            with open(
                f"{self._temp_dir}sequences_0.json", "r", encoding="utf-8"
            ) as file:
                hs_sequences = json.load(file)
        except FileNotFoundError:
            matrices = np.eye(2, dtype=complex).reshape(1, 2, 2)
            sequences = [""]
            duplicates = {}
            for length in range(1, 8):
                for seq in product(self.clifford_gates, repeat=length):
                    seqstr = _substitute_duplicates("".join(seq), duplicates)
                    if seqstr in sequences:
                        continue
                    matrix = seq2mat(seqstr)
                    duplicate = False
                    for existing_seq, existing_mat in zip(sequences, matrices):
                        if not np.allclose(trace(existing_mat, matrix), 1):
                            continue
                        duplicate = True
                        counts = []
                        for s in (existing_seq, seqstr):
                            counts.append([])
                            for gate in self.nontrivial_clifford_gates:
                                counts[-1].append(s.count(gate))
                            counts[-1].append(len(s))
                            counts[-1].append(s)
                        if tuple(counts[0]) > tuple(counts[1]):
                            duplicates[existing_seq] = seqstr
                        else:
                            duplicates[seqstr] = existing_seq
                        break
                    if not duplicate:
                        sequences.append(seqstr)
                        matrices = np.vstack([matrices, matrix.reshape(1, 2, 2)])

            matrices = matrices.transpose(1, 0, 2)
            np.save(f"{self._temp_dir}tensor_0.npy", matrices)
            with open(
                f"{self._temp_dir}sequences_0.json", "w", encoding="utf-8"
            ) as file:
                json.dump(sequences, file, indent=4)
            with open(
                f"{self._temp_dir}duplicates_0.json", "w", encoding="utf-8"
            ) as file:
                json.dump(duplicates, file, indent=4)
            hs_sequences = sequences

        t_block = np.einsum("ij,jpk->pik", t(), matrices)
        sqrt_t_block = np.einsum("ij,jpk->pik", sqrt_t(), matrices)

        for length in range(1, self.max_len + 1):
            with open(
                f"{self._temp_dir}duplicates_{length - 1}.json",
                "r",
                encoding="utf-8",
            ) as file:
                duplicates = json.load(file)
            with open(
                f"{self._temp_dir}sequences_{length - 1}.json", "r", encoding="utf-8"
            ) as file:
                sequences = json.load(file)
            matrices = np.load(f"{self._temp_dir}tensor_{length - 1}.npy").transpose(
                1, 0, 2
            )
            matrices = cp.asarray(matrices)
            t_block = cp.asarray(t_block)
            sqrt_t_block = cp.asarray(sqrt_t_block)
            counter = 0

            for (mat1, seq1), (mat2, seq2, nc_symb) in product(
                zip(matrices, sequences),
                chain(
                    zip(t_block, hs_sequences, ["T"] * len(hs_sequences)),
                    zip(sqrt_t_block, hs_sequences, ["sqrtT"] * len(hs_sequences)),
                ),
            ):
                if nc_symb == "T":
                    seqstr = _substitute_duplicates(seq1 + "t" + seq2, duplicates)
                else:
                    seqstr = _substitute_duplicates(seq1 + "q" + seq2, duplicates)

                counter += 1
                if counter % 1000 == 0:
                    print(
                        f"number of t/q gates {length}, iteration counter: {counter}, number of unique sequences: {len(sequences)}"
                    )
                if seqstr in sequences:
                    continue
                matrix = mat1 @ mat2
                duplicate = cp.argwhere(
                    cp.isclose(
                        cp.abs(
                            cp.dot(matrices[:, 0], matrix[0].conj())
                            + cp.dot(matrices[:, 1], matrix[1].conj())
                        ),  # calculate trace without matrix multiplication
                        2,
                    )
                )
                if len(duplicate) == 0:
                    sequences.append(seqstr)
                    matrices = cp.vstack([matrices, matrix.reshape(1, 2, 2)])
                else:
                    existing_seq = sequences[duplicate[0][0].item()]
                    counts = []
                    for s in (existing_seq, seqstr):
                        counts.append([])
                        for gate in (
                            self.non_clifford_gates + self.nontrivial_clifford_gates
                        ):
                            counts[-1].append(s.count(gate))
                        counts[-1].append(len(s))
                        counts[-1].append(s)
                    if tuple(counts[0]) > tuple(counts[1]):
                        duplicates[existing_seq] = seqstr
                    elif existing_seq != seqstr:
                        duplicates[seqstr] = existing_seq

            np.save(
                f"{self._temp_dir}tensor_{length}.npy",
                asnumpy(matrices.transpose(1, 0, 2)),
            )
            with open(
                f"{self._temp_dir}sequences_{length}.json", "w", encoding="utf-8"
            ) as file:
                json.dump(sequences, file, indent=4)
            with open(
                f"{self._temp_dir}duplicates_{length}.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(duplicates, file, indent=4)


if __name__ == "__main__":
    creator = SequenceCreator()
    creator.generate_unique_sequences()
