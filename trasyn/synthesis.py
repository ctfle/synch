import json
import warnings
from dataclasses import dataclass
from functools import cached_property
from itertools import product, permutations
from typing import Literal, Iterable

from abc import abstractmethod, ABC
from trasyn.utils import (
    can_partition_with_multiples,
    distance,
    seq2mat,
    tuple_partitions,
    find_unsupported_trajectories,
    backtrack_sequences_with_cost,
)

import numpy as np
from numpy.typing import NDArray
import os

try:
    import cupy as cp
    from cupy_backends.cuda.api.runtime import CUDARuntimeError

    asnumpy = cp.asnumpy
    MemError = CUDARuntimeError
except ModuleNotFoundError:
    cp = np
    asnumpy = np.asarray
    MemError = MemoryError

from trasyn.mps import _sample, _trace_target_unitary
from trasyn.utils import get_available_memory

ASSETS_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/assets"
AVAILABLE_GATE_SETS = [
    dir_entry
    for dir_entry in sorted(os.listdir(ASSETS_DIR))
    if os.path.isdir(path := f"{ASSETS_DIR}/{dir_entry}")
]

seed = 42
rng = np.random.default_rng(seed=seed)


def _num_candidates(
    nonclifford_count: int | NDArray[np.int64], nonclifford_gate: Literal["t"] = "t"
) -> int | NDArray[np.int64]:
    if nonclifford_gate != "t":
        raise NotImplementedError(
            f"Non-Clifford gate {nonclifford_gate} is not supported yet."
        )
    return np.clip(72 * 2.0**nonclifford_count - 48, 0, None).astype(np.int64)


def _substitute_duplicates(target_sequence: str, lookup_table: dict[str, str]) -> str:
    old_sequence = ""
    while old_sequence != target_sequence:
        old_sequence = target_sequence
        for key, value in lookup_table.items():
            target_sequence = target_sequence.replace(key, value)
    return target_sequence


@dataclass
class SynthesisResult:
    seqstr: str
    error: float
    target_unitary: np.ndarray


class BudgetPartitioner(ABC):
    def __init__(
        self,
        costs: dict[str, float],
        total_non_clifford_budget: float,
        max_partition_value: float,
    ):
        self.costs = costs
        self.total_non_clifford_budget = total_non_clifford_budget
        self.max_partition_value = max_partition_value
        self._verify_costs()

    def _verify_costs(self):
        assert np.allclose(self.min_gate_cost, 1), "min cost should be set to 1."
        for id, cost in self.costs.items():
            assert float(cost).is_integer() or (cost - 0.5).is_integer(), (
                "Costs must be integer or multiple of 1/2."
            )
            assert len(id) == 1, "Cost id must be single char. "

    @property
    def cost_values(self) -> list[float]:
        return list(self.costs.values())

    @cached_property
    def max_gate_cost(self) -> float:
        return max([cost for cost in self.costs.values()])

    @cached_property
    def min_gate_cost(self) -> float:
        return min([cost for cost in self.costs.values()])

    @property
    def all_costs_integer(self) -> bool:
        return all([float(i).is_integer() for i in self.costs.values()])

    @abstractmethod
    def partition(self) -> list[list[int | float]]:
        """Implements the partitioning."""


class FilteredBudgetPartitioner(BudgetPartitioner):
    """
    Partitioner that partitions into integers with at most one non-integer component.
    """

    def partition(self) -> list[list[float | int]]:
        if self.all_costs_integer:
            budgets = [
                self._get_integer_partition(i)
                for i in range(int(self.total_non_clifford_budget + 1))
            ]
        else:
            budgets = []
            for i in np.arange(0, self.total_non_clifford_budget + 0.5, 0.5):
                partition = self._get_non_integer_partition(i)
                if len(partition) > 0:
                    budgets.append(partition)

        budgets = self._minimize_partitioning(budgets)
        self._add_permutations(budgets)
        self._verify_partition(budgets)

        return budgets

    def _get_integer_partition(self, input: int) -> list[int]:
        # if the total non_clifford budget is int we can call the normal partitioner
        # partition using as many 2 * max_gate_cost as possible
        if input < self.max_partition_value:
            partition = [input]
        else:
            partition = self._complete_partitioning(input)
        return partition

    def _get_non_integer_partition(self, input: float):
        # iterate in steps of 0.5
        # for each value, check if
        partition = []
        if input < self.max_partition_value:
            if can_partition_with_multiples(input, self.cost_values):
                partition = [input]
        else:
            if input.is_integer():
                # even number of .5 numbers must be used
                # split of max_cost *2
                partition = self._complete_partitioning(input)
            else:
                # odd number of .5 numbers must be used
                # split off max_cost
                partition = [self.max_gate_cost]
                partition.extend(
                    self._complete_partitioning(input - self.max_gate_cost)
                )

        return partition

    def _complete_partitioning(self, remainder: float):
        partition = []
        while remainder > 2 * self.max_gate_cost:
            partition.append(2 * self.max_gate_cost)
            remainder -= 2 * self.max_gate_cost

        partition.append(remainder)
        return partition

    def _minimize_partitioning(
        self, budgets: list[list[int | float]]
    ) -> list[list[int | float]]:
        """Minimizes the number of partitions in each element of the input list."""
        new_budgets_partitioning = []
        for budget_partition in budgets:
            new_budget = [budget_partition[0]]
            for part in budget_partition[1:]:
                ind = self._adds_to(new_budget, part)
                if ind is not None:
                    new_budget[ind] += part
                else:
                    new_budget.append(part)

            new_budgets_partitioning.append(new_budget)
        return new_budgets_partitioning

    def _add_permutations(
        self, budgets: list[list[int | float]]
    ) -> list[list[int | float]]:
        """Add in all the permutations and drop duplicates."""
        perms = [
            list(permutation)
            for budget in budgets
            for permutation in permutations(budget)
        ]

        # drop duplicates
        seen = set()
        out = []
        for sub in perms:
            t = tuple(sub)
            if t not in seen:
                seen.add(t)
                out.append(sub)

        return out

    def _verify_partition(self, budgets: list[list[int | float]]):
        for budget in budgets:
            if len(budget) == 1:
                continue
            non_integer_parts = 0
            for part in budget:
                if not float(part).is_integer():
                    non_integer_parts += 1
            assert non_integer_parts <= 1

    def _adds_to(
        self, budget_partitioning: list[int | float], value: int | float
    ) -> int | None:
        """
        Looks into budget_partitioning and checks if value can be added to any element
        so that this element is still < self.max_partition_value. Returns the index of the
        corresponding element or None otherwise.
        """
        index = None
        for i, budget in enumerate(budget_partitioning):
            if value + budget <= self.max_partition_value:
                index = i
                break
        return index


class ErgodicPartitioner(BudgetPartitioner):
    """
    Takes responsibility of creating a suitable partitioning.
    The partitioning has to be done with care such that every possible seauence can actually be
    reached when sampling. This is especially important for cases where the costs dict contains
    entries which differ in cost.
    """

    def partition(self) -> list[list[int | float]]:
        if self.all_costs_integer:
            budgets = []
            for i in range(int(self.total_non_clifford_budget + 1)):
                budgets.extend(self._get_ergodic_integer_partition(i))
        else:
            budgets = []
            for i in np.arange(0, self.total_non_clifford_budget + 0.5, 0.5):
                partition = self._get_ergodic_non_integer_partition(i)
                if len(partition) > 0:
                    budgets.extend(partition)

        return budgets

    def _get_ergodic_integer_partition(self, total_cost):
        if total_cost < self.max_partition_value:
            return [[total_cost]]
        else:
            min_subset = None
            length = 1
            while min_subset is None:
                length += 1
                raw_partitions = tuple_partitions(
                    total_cost, 1, self.max_partition_value, length=length, step=1
                )

                _, min_subset = self._greedy_min_subset(
                    raw_partitions
                )

            return list(map(list, min_subset))

    def _get_ergodic_non_integer_partition(self, total_cost):
        partition = []
        if total_cost < self.max_partition_value:
            if can_partition_with_multiples(total_cost, self.cost_values):
                partition = [[total_cost]]
        else:
            min_subset = None
            length = 1
            while min_subset is None:
                length += 1
                raw_partitions = tuple_partitions(
                    total_cost, 1, self.max_partition_value, length=length, step=0.5
                )
                _, min_subset = self._greedy_min_subset(
                    raw_partitions
                )

            partition = list(map(list, min_subset))
        return partition

    def _is_valid_partition(self, partition):
        return find_unsupported_trajectories(self.costs, partition) == []

    def _greedy_min_subset(self, elements: list[list[int | float ]]):
        remaining = list(elements)
        subset = []
        while remaining:
            # Score by marginal gain: f(subset + [e]) improvement
            scores = []
            current_remainder = self._evaluate_remainder(subset, elements)
            for e in remaining:

                test_subset = subset + [e]
                if current_remainder > len(find_unsupported_trajectories(self.costs, test_subset)):
                    scores.append((1.0 / (len(test_subset) + 1), e))  # Favor smaller
                else:
                    scores.append((0, e))

            if not any(s[0] > 0 for s in scores):
                break

            # Pick best
            _, best_e = max(scores)
            subset.append(best_e)
            remaining.remove(best_e)

        return (len(subset), set(subset)) if find_unsupported_trajectories(self.costs, subset) == [] else (None, None)
    
    def _evaluate_remainder(self, subset: list[list[int | float]], elements: list[list[int | float]]):
        total_cost = sum(elements[0])
        return len(find_unsupported_trajectories(self.costs, subset)) if subset else len(
            backtrack_sequences_with_cost(total_cost, self.costs))




class Synthesiser:
    def __init__(
        self,
        partitioner: BudgetPartitioner,
        error_threshold: float | None = None,
        load_dir: str = f"{ASSETS_DIR}" + "tshxyz",
        num_attempts: int = 5,
        num_samples: int | None = None,
    ):
        self.load_dir = load_dir
        self.error_threshold = error_threshold
        self._num_samples = num_samples
        self.num_attempts = num_attempts
        self.budget_partitioner = partitioner

    @property
    def mem_size(self):
        return get_available_memory(gpu=False)

    def get_num_samples(self, mps: list[NDArray], budget: list[int]) -> int:
        if self._num_samples is None:
            if len(mps) == 1:
                return 1
            else:
                return self.mem_size // (
                    max(tsr.shape[1] * tsr.shape[2] for tsr in mps[1:])
                    * 2 ** (4 + len(budget))
                )
        else:
            return self._num_samples

    @property
    def budget_composition(self) -> list[list[int]] | list[list[float]]:
        return self.budget_partitioner.partition()

    def get_tensor(self, budget: int):
        """
        Loads a tensor with a given non-clifford budget. Shape is (2, N, 2) where N is the number
        of different 2x2 matrices given the fixed non-clifford budget.
        """
        return np.load(self.load_dir + "/" + f"tensor_{budget:.1f}.npy")

    def get_tensor_as_str(self, budget: float) -> list[str]:
        """
        Loads the sequence of gates associated with given budget.
        """
        # load sequence strings
        with open(
            self.load_dir + "/" + f"sequences_{budget:.1f}.json",
            "r",
            encoding="utf-8",
        ) as f:
            sequences = json.load(f)

        return sequences

    def get_duplicate_as_str(self, budget: float) -> dict[str, str]:
        """
        Loads the duplicates of gate sequences associated with given budget.
        """
        # load sequence strings
        with open(
            self.load_dir + "/" + f"duplicates_{budget:.1f}.json",
            "r",
            encoding="utf-8",
        ) as f:
            duplicates = json.load(f)

        return duplicates

    def get_sequence_of_tensors(self, budget: list[int]) -> list[NDArray]:
        """Generate a list of corresponding tensors according to budget."""
        return [self.get_tensor(b) for b in budget]

    def get_sequence_of_tensors_as_str(self, budget: Iterable[int]) -> list[list[str]]:
        return [self.get_tensor_as_str(b) for b in budget]

    def get_duplicates(self, budget: list[int]) -> Iterable[dict[str, str]]:
        return [self.get_duplicate_as_str(b) for b in budget]

    def sample_and_synthesize(
        self, target_unitary: NDArray, verbose: bool
    ) -> SynthesisResult:
        fidelity = 0
        bitstring = None
        result = SynthesisResult(error=2, seqstr="", target_unitary=target_unitary)
        for budget, _ in product(self.budget_composition, range(self.num_attempts)):
            mps = self.get_sequence_of_tensors(budget)
            mps = _trace_target_unitary(mps, target_unitary)
            n_samples = self.get_num_samples(mps, budget)
            while n_samples:
                try:
                    bitstring, fidelity = _sample(mps, n_samples, rng=rng)
                    break
                except MemError:
                    n_samples = int(n_samples * 0.9)

            fidelity /= 2
            # TODO: this makes no sense. Fidelity should not be > 1
            fidelity = min(fidelity, 1)
            error = np.sqrt(1 - fidelity**2)
            if verbose:
                print(f"Budget: {budget}, Num samples: {n_samples}")
                print(f"Error:{error}, Fidelity: {fidelity}")
            if error < result.error:
                result = SynthesisResult(
                    error=error,
                    seqstr=self.get_sequence_str(bitstring, budget),
                    target_unitary=target_unitary,
                )
            if self.error_threshold is not None and error <= self.error_threshold:
                break

        self._verify_result(target_unitary, result)
        return result

    def _verify_result(self, target_unitary: NDArray, result: SynthesisResult):
        if self.error_threshold is not None and result.error > self.error_threshold:
            warnings.warn(
                f"Error threshold {self.error_threshold} is not reached "
                f"by the lowest error found: {result.error}."
            )
        assert np.allclose(
            distance(target_unitary, seq2mat(result.seqstr)), result.error
        )

    def get_sequence_str(self, indices: Iterable[int], budget: Iterable[int]) -> str:
        """Get the sequences of gates as str associated with the budget and the indices"""
        tensors_as_string = self.get_sequence_of_tensors_as_str(budget)
        duplicates = self.get_duplicates(budget)
        seqstr = []
        for index, tensor, dupl in zip(indices, tensors_as_string, duplicates):
            target_str = _substitute_duplicates(tensor[index], dupl)
            seqstr.append(target_str)

        return "".join(seqstr)
