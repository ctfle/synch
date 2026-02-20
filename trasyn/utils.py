import json
import os
import subprocess
from collections.abc import Generator, Sequence
from functools import lru_cache
from typing import Callable

import numpy as np
import psutil
from numpy.typing import NDArray

from trasyn.gates import GATES

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