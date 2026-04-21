import json
import os
import subprocess
from collections.abc import Generator, Sequence
from copy import copy
from functools import lru_cache
from itertools import combinations
from typing import Callable

import numpy as np
import psutil
from numpy.typing import NDArray

from trasyn.src.gates import GATES

MAX_CACHE_LEN = 8


@lru_cache(maxsize=3**MAX_CACHE_LEN)
def _seq2mat_cache(gate_seq: str) -> NDArray[np.complex128]:
    length = len(gate_seq)
    if length == 0:
        return np.eye(2)
    if length == 1:
        try:
            return GATES[gate_seq.lower()]()
        except KeyError as err:
            raise ValueError(
                f"Unknown gate: {gate_seq}. Available gates: {', '.join(GATES)}."
            ) from err
    return _seq2mat_cache(gate_seq[: length // 2]) @ _seq2mat_cache(
        gate_seq[length // 2 :]
    )


def seq2mat(gate_seq: str) -> NDArray[np.complex128]:
    if len(gate_seq) > MAX_CACHE_LEN:
        return _seq2mat_cache(gate_seq[:MAX_CACHE_LEN]) @ seq2mat(
            gate_seq[MAX_CACHE_LEN:]
        )
    return _seq2mat_cache(gate_seq)


def trace(mat1: NDArray[np.complex128], mat2: NDArray[np.complex128]) -> float:
    return min(np.abs(np.trace(mat1 @ mat2.conj().T) / mat1.shape[0]), 1)


def distance(mat1: NDArray[np.complex128], mat2: NDArray[np.complex128]) -> float:
    return np.sqrt(1 - trace(mat1, mat2) ** 2)


def transpose(
    system: NDArray[np.complex128], first_qubit: int, second_qubit: int
) -> NDArray[np.complex128]:
    if first_qubit == second_qubit:
        return system
    num_qubits = int(np.log2(system.shape[0]))
    source, destination = [first_qubit], [second_qubit]
    original_shape, tsr_shape = system.shape, [2] * num_qubits
    if system.ndim > 1:
        source.append(first_qubit + num_qubits)
        destination.append(second_qubit + num_qubits)
        tsr_shape = [2] * (2 * num_qubits)
    return np.moveaxis(system.reshape(tsr_shape), source, destination).reshape(
        original_shape
    )


def find_file_to_read(
    base_dir: str | Sequence[str],
    exclude_keywords: Sequence[str] = (),
    exclude_filenames: Sequence[str] = (),
) -> Generator[str, None, None]:
    if isinstance(base_dir, str):
        base_dir = [base_dir]
    for directory in base_dir:
        if directory.endswith("/"):
            directory = directory[:-1]
        for filename in sorted(os.listdir(directory)):
            if filename in exclude_filenames or any(
                k in filename for k in exclude_keywords
            ):
                continue
            if os.path.isdir(path := f"{directory}/{filename}"):
                yield from find_file_to_read(path, exclude_keywords, exclude_filenames)
            yield path


def find_json_to_read(
    base_dir: str | Sequence[str],
    exclude_keywords: Sequence[str] = (),
    exclude_filenames: Sequence[str] = (),
) -> Generator[tuple[str, dict], None, None]:
    for path in find_file_to_read(base_dir, exclude_keywords, exclude_filenames):
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            yield path.split("/")[-1][:-5].lower().replace("_", " "), data


def can_partition_with_multiples(target: float, numbers: list[float], tol=1e-8) -> bool:
    """
    Check if target can be made using any combination of numbers (multiples allowed).

    Examples:
    >>> can_partition_with_multiples(8.5, [4.0, 2.5, 1.0])  # True: 4.0 + 4.0 + 0.5
    >>> can_partition_with_multiples(7.5, [3.0, 2.5])       # True: 2.5 + 2.5 + 2.5
    """
    # Scale to integers (multiply by 2 for half-integers)
    target_scaled = round(target * 2)
    numbers_scaled = [round(n * 2) for n in numbers if abs(n) > tol]

    target_int = int(target_scaled)

    # dp[s] = True if sum s can be made
    dp = [False] * (target_int + 1)
    dp[0] = True  # sum 0 always possible

    # Try each number for each sum
    for s in range(target_int + 1):
        if dp[s]:
            for num in numbers_scaled:
                if s + num <= target_int:
                    dp[s + num] = True

    return dp[target_int]


def find_index(lst: list, pred: Callable) -> list:
    return [i for i, x in enumerate(lst) if pred(x)]


def get_available_memory(gpu: bool = False) -> int:
    if gpu:
        memsize, unit = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"]
            )
            .decode()
            .strip()
            .split()
        )
        memsize = int(memsize)
        if unit == "MiB":
            memsize *= 1024**2
        elif unit == "GiB":
            memsize *= 1024**3
        else:
            raise NotImplementedError(
                f"Unknown memory unit {unit}. Only MiB and GiB are supported at this moment."
            )
    else:
        memsize = psutil.virtual_memory().available
    return memsize


try:
    from qiskit import QuantumCircuit, qasm2

    def find_qasm_to_read(
        base_dir: str | Sequence[str],
        exclude_keywords: Sequence[str] = (),
        exclude_filenames: Sequence[str] = (),
    ) -> Generator[tuple[str, QuantumCircuit], None, None]:
        for path in find_file_to_read(base_dir, exclude_keywords, exclude_filenames):
            if (filename := path.split("/")[-1]).endswith(".qasm"):
                yield (
                    filename[:-5].lower().replace("_", " "),
                    qasm2.load(
                        path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
                    ),
                )

except ImportError:
    pass


def random_unitary_2x2():
    # complex Ginibre matrix
    Z = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2)

    # QR decomposition
    Q, R = np.linalg.qr(Z)

    # fix phases to ensure uniform Haar measure
    D = np.diag(R)
    Q = Q * (D / np.abs(D))

    return Q


def count_t_equiv(seqstr):
    """Count T-equivalent gates: t=1, q=2, cliffords=0"""
    t_count = seqstr.count("t")
    q_count = seqstr.count("q")
    return t_count + 2.5 * q_count


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
        if len(value) < len(key):
            new_dupl[key] = value
        if len(value) > len(key):
            new_dupl[value] = key
        if len(value) == len(key):
            continue

    return new_dupl


def replace(sequences: list[str], old: str, new: str) -> list[str]:
    for i, seq in enumerate(sequences):
        sequences[i] = seq.replace(old, new)

    return sequences


def tuple_partitions(
    n: float,
    lower_bound: float,
    upper_bound: float,
    step: float,
    length: int,
    tol: float = 1e-9,
):
    """
    Return all tuples of given length whose elements sum to n.
    Each element lies within [lower_bound, upper_bound] and is an integer multiple of step.
    """

    # Discretize allowed values
    values = [
        round(lower_bound + i * step, 10)
        for i in range(int((upper_bound - lower_bound) / step) + 1)
    ]
    results = []

    # Efficient iterative filtering rather than brute-force product
    def search(prefix, remaining_sum, depth):
        if depth == length - 1:
            last = remaining_sum
            # Check if last fits the discretization grid and bounds
            if (lower_bound - tol) <= last <= (upper_bound + tol):
                if abs(last / step - round(last / step)) < tol:
                    results.append(tuple(prefix + [round(last, 10)]))
            return

        for v in values:
            # Prune if overshooting is obvious
            min_possible = (length - depth - 1) * lower_bound
            max_possible = (length - depth - 1) * upper_bound
            if not (min_possible - tol <= remaining_sum - v <= max_possible + tol):
                continue
            search(prefix + [v], remaining_sum - v, depth + 1)

    search([], n, 0)
    return results


def min_valid_subset(original_set, f):
    elements = list(original_set)
    n = len(elements)

    best_subset = None
    best_size = 0

    def has_valid_subset(k):
        nonlocal best_subset, best_size
        if k == 0:
            if f(set()):
                best_subset = set()
                best_size = 0
            return f(set())

        for subset_tuple in combinations(elements, k):
            subset = set(subset_tuple)
            if f(subset):
                # Found valid! Update best
                if k < best_size or best_size == 0:
                    best_subset = subset
                    best_size = k
                return True  # Early exit: size k possible
        return False

    # Binary search for min size
    low, high = 0, n
    while low < high:
        mid = (low + high + 1) // 2
        if has_valid_subset(mid):
            high = mid - 1
        else:
            low = mid
            # high = mid - 1

    # Verify final result
    if best_size > 0:
        return best_size, best_subset

    return None, None  # No valid subset exists


def backtrack_sequences_with_cost(
    target: float, costs: dict[str, float | int]
) -> list[str]:
    """
    Creates all possible combinations of identifiers (keys of cost) so that the sum of the
    costs gives the target.
    """
    result: list[str] = []
    eps = 1e-9  # to handle float rounding

    def backtrack(current: str, total: float):
        if abs(total - target) < eps:
            result.append(current)
            return
        if total > target + eps:
            return

        for char, cost_value in costs.items():
            backtrack(current + char, total + cost_value)

    backtrack("", 0.0)
    return result


def find_unsupported_trajectories(
    costs: dict[str, int | float], partitioning: list[list[int | float]]
):
    """
    Find all the trajectories i.e. sequences of gates in the cost dict that are not supported by
    the provided partitioning.
    For Example: there are t and q gates in the costs dict, we built all the sequences of only t
    and q according to the total costs extracted from the partitioning.
    costs = {t:1.0, q:2.5}, from partitioning the total cost is 5
    sequences = [qq, ttttt]
    loop through the partitionings and check which of the above sequences is supported and which not
    """
    targets = list(map(sum, partitioning))
    assert all(x == targets[0] for x in targets), (
        "total cost must be equal for all input partitionings."
    )

    sequences = backtrack_sequences_with_cost(targets[0], costs)

    unsupported = []
    for seq in sequences:
        budgets = [copy(part) for part in partitioning]
        if not any([supports_sequence(budget, costs, seq) for budget in budgets]):
            unsupported.append(seq)

    return unsupported


def supports_sequence(
    budget: list[int | float], costs: dict[str, int | float], sequence: str
):
    tol = 1e-8
    current_cost = 0.0
    current_index = 0
    current_cost_ceiling = budget[current_index]
    for char in sequence:
        if abs(current_cost_ceiling - current_cost) < tol:
            current_index += 1
            current_cost_ceiling = budget[current_index]
            current_cost = 0.0

        if (
            costs[char] - (current_cost_ceiling - current_cost) > tol
            and abs(current_cost_ceiling - current_cost) > tol
        ):
            # char doesn't fit in current bucket but bucket is not full yet
            # means sequence is not supported
            return False

        # char fits in current bucket
        if costs[char] <= current_cost_ceiling - current_cost:
            current_cost += costs[char]

    return True
