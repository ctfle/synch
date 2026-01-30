import json
import os
import warnings
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from math import log
from multiprocessing import Pool
from typing import Literal, Sequence, Iterable
from tqdm import tqdm
from trasyn.utils import can_partition_with_multiples

import numpy as np
# from hypothesis.internal.conjecture.shrinking import Collection
from numpy.random import Generator
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from sympy.physics.quantum.density import fidelity



import os
import pickle
from pathlib import Path

try:
    import cupy as cp
    from cupy_backends.cuda.api.runtime import CUDARuntimeError

    asnumpy = cp.asnumpy
    MemError = CUDARuntimeError
except ModuleNotFoundError:
    cp = np
    asnumpy = np.asarray
    MemError = MemoryError

from trasyn.gates import rz, t, u
from trasyn.mps import _sample, _trace_target_unitary
from trasyn.utils import distance, get_available_memory, seq2mat

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
        raise NotImplementedError(f"Non-Clifford gate {nonclifford_gate} is not supported yet.")
    return np.clip(72 * 2.0 ** nonclifford_count - 48, 0, None).astype(np.int64)


def _substitute_duplicates(target_sequence: str, lookup_table: dict[str, str]) -> str:
    old_sequence = ""
    while old_sequence != target_sequence:
        old_sequence = target_sequence
        for key, value in lookup_table.items():
            target_sequence = target_sequence.replace(key, value)
    return target_sequence


@dataclass
class SynthesisResult():
    seqstr: str
    error: float



class BudgetPartitioner():

    """ Not sure about the factor of 2 """

    def __init__(self, costs: dict[str, float],
                 total_non_clifford_budget: float,
                 max_partition_value: float):
        self.costs = costs
        self.total_non_clifford_budget = total_non_clifford_budget
        self.max_partition_value = max_partition_value
        self._verify_costs()
        self._verify_max_partition_value()

    def _verify_costs(self):
        assert np.allclose(self.min_gate_cost, 1), "min cost should be set to 1."
        for cost in self.costs.values():
            assert cost.is_integer() or (cost - 0.5).is_integer(), "Costs must be integer or multiple of 1/2."

    def _verify_max_partition_value(self):
        assert self.max_partition_value >= self.max_gate_cost * 2, f"max partition value must be > 2 * gate cost value {self.max_gate_cost}"

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
        return all([i.is_integer() for i in self.costs.values()])

    def partition(self) -> list[list[float]] | list[list[int]]:
        if self.all_costs_integer:
            budgets = [self._get_integer_partition(i)
                       for i in range(int(self.total_non_clifford_budget + 1))]
        else:
            budgets = []
            for i in np.arange(0, self.total_non_clifford_budget + 0.5, 0.5):
                partition = self._get_non_integer_partition(i)
                if len(partition) > 0:
                    budgets.append(partition)

        return budgets

    def _get_integer_partition(self, input: int) -> list[int]:
        # if the total non_clifford budget is int we can call the normal partitioner
        partition = []
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
                partition.extend(self._complete_partitioning(input - self.max_gate_cost))

        return partition

    def _complete_partitioning(self, remainder: float):
        partition = []
        while remainder > 2 * self.max_gate_cost:
            partition.append(2 * self.max_gate_cost)
            remainder -= 2 * self.max_gate_cost

        partition.append(remainder)
        return partition

    def _original_partitioning(self):
        budgets = [[curr_budget + 1] for curr_budget in range(int(min(self.total_non_clifford_budget, self.max_partition_value)))]
        for curr_budget in range(int(self.max_partition_value + 1), int(self.total_non_clifford_budget + 1)):
            # compute how many tensors we need. in this for loop the first value is 2
            num_tensors =int( (curr_budget - 1) // self.max_partition_value + 1 )
            #print(curr_budget, num_tensors)
            # compute the floor i.e. instead of [5] append [2, 2]
            budget_decomposition = [curr_budget // num_tensors] * num_tensors
            #print(budget_decomposition)
            # eventually we want that the sum of the elements of this list is exactly current budget
            # in the example above we turned [5] into [2,2] but sum([2,2]) = 4
            # to correct for this, we pick the fist element of the and replace it with the correct
            # value which is curr_budget -sum(all but the first element)

            # now it can happen that the first element in the budget_decomposition is > max_count
            # in this case we need to distribute it to other elements
            if curr_budget - sum(budget_decomposition[1:]) <= self.max_partition_value:
                budget_decomposition[0] = curr_budget - sum(budget_decomposition[1:])
            else:
                budget_element = curr_budget - sum(budget_decomposition[1:])
                index = 1
                while budget_element > self.max_partition_value and index < len(budget_decomposition):
                    budget_decomposition[index] += 1
                    index += 1
                    budget_element -= 1

                if budget_element > self.max_partition_value:
                    raise NotImplementedError("Other stratgey required")
                budget_decomposition[0] = budget_element

            budgets.append(budget_decomposition)

        return budgets



class Sythesiser():

    def __init__(self,
        partitioner: BudgetPartitioner,
        error_threshold: float | None = None,
        gate_set: str = "tshxyz",
        num_attempts: int = 5,
        num_samples: int | None = None,):

        self.gate_set = gate_set.lower()
        self.error_threshold = error_threshold
        self._num_samples = num_samples
        self.num_attempts = num_attempts
        self.budget_partitioner = partitioner
        self.integer_budgets: bool = True

    @property
    def mem_size(self):
        return get_available_memory(gpu=False)

    def get_num_samples(self, mps: list[NDArray], budget: list[int]) -> int:
        if self._num_samples is None:
            if len(mps) == 1:
                return 1
            else:
                return self.mem_size // (
                        max(tsr.shape[1] * tsr.shape[2] for tsr in mps[1:]) * 2 ** (
                        4 + len(budget))
                )
        else:
            return self._num_samples

    @property
    def load_dir(self):
        return f"{ASSETS_DIR}/{self.gate_set}/"

    @property
    def budget_composition(self) -> list[list[int]] | list[list[float]]:
        """ Computes a list of lists where each element yields a possible composition of ints that
         sum to the index + 1 of this element in the list. Example:
         [[1], [2], [3], [4], [3,2], [3,3]]
        Up to max count we can use list with a single entry. Then we need to do combinations.
         """
        return self.budget_partitioner.partition()

    def get_tensor(self, budget: int):
        """
        Loads a tensor with a given non-clifford budget. Shape is (2, N, 2) where N is the number
        of different 2x2 matrices given the fixed non-clifford budget.
        """
        return np.load(self.load_dir + f"tensor_{budget:.1f}.npy")

    def get_tensor_as_str(self, budget: float) -> list[str]:
        """
        Loads the sequence of gates associated with given budget.
        """
        # load sequence strings
        with open(
                self.load_dir +
                f"sequences_{budget:.1f}.json",
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
                self.load_dir +
                f"duplicates_{budget:.1f}.json",
                "r",
                encoding="utf-8",
        ) as f:
            duplicates = json.load(f)

        return duplicates

    def get_sequence_of_tensors(self, budget: list[int]) -> list[NDArray]:
        """ Generate a list of corresponding tensors according to budget. """
        return [self.get_tensor(b) for b in budget]

    def get_sequence_of_tensors_as_str(self, budget: Iterable[int]) -> list[list[str]]:
        return [self.get_tensor_as_str(b) for b in budget]

    def get_duplicates(self, budget: list[int]) -> Iterable[dict[str, str]]:
        return [self.get_duplicate_as_str(b) for b in budget]

    def sample_and_synthesize(self, target_unitary: NDArray, verbose: bool) -> SynthesisResult:
        fidelity = 0
        bitstring = None
        result = SynthesisResult(error=2, seqstr="")
        for budget, _ in product(self.budget_composition, range(self.num_attempts)):
            #print(budget)
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
            error = np.sqrt(1 - fidelity ** 2)
            if verbose:
                print(f"Budget: {budget}, Num samples: {n_samples}")
                print(f"Error:{error}, Fidelity: {fidelity}")
            if error < result.error:
                result = SynthesisResult(error=error, seqstr=self.get_sequence_str(bitstring, budget))
            if self.error_threshold is not None and error <= self.error_threshold:
                break

        if self.error_threshold is not None and result.error > self.error_threshold:
            warnings.warn(
                f"Error threshold {self.error_threshold} is not reached "
                f"by the lowest error found: {result.error}."
            )

        return result

    def get_sequence_str(self, indices: Iterable[int], budget: Iterable[int]) -> str:
        """ Get the sequences of gates as str associated with the budget and the indices """
        tensors_as_string = self.get_sequence_of_tensors_as_str(budget)
        duplicates = self.get_duplicates(budget)
        seqstr = []
        for index, tensor, dupl in zip(indices, tensors_as_string, duplicates):
            target_str = _substitute_duplicates(tensor[index], dupl)
            seqstr.append(target_str)

        return "".join(seqstr)


def random_unitary_2x2():
    # complex Ginibre matrix
    Z = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2)

    # QR decomposition
    Q, R = np.linalg.qr(Z)

    # fix phases to ensure uniform Haar measure
    D = np.diag(R)
    Q = Q * (D / np.abs(D))

    return Q


def benchmark_on_random_unitaries(n_unitaries_per_budget: int,
                                budgets: list[float] | list[int],
                                gate_set: str,
                                costs: dict[str, float],
                                save_dir: str = "./benchmark_results"):
    # Ensure save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    error_data = {}
    seqstr_data = {}
    for nc_budget in tqdm(budgets):
        partitioner = BudgetPartitioner(max_partition_value=5,
                                        total_non_clifford_budget=nc_budget,
                                        costs=costs)
        syn = Sythesiser(partitioner=partitioner,
                               gate_set=gate_set)

        errors = []
        seqs = []
        for i in range(n_unitaries_per_budget):
            target_unitary = random_unitary_2x2()
            result = syn.sample_and_synthesize(target_unitary, verbose=False)
            errors.append(result.error)
            seqs.append(result.seqstr)

        error_data[nc_budget] = errors
        seqstr_data[nc_budget] = seqs
        # Save per-budget checkpoint
        budget_data = {
            'error_data': error_data,
            'seqstr_data': seqstr_data,
            'budgets': budgets,
            'n_unitaries_per_budget': n_unitaries_per_budget
        }
        pickle.dump(budget_data,
                    open(f"{save_dir}/{gate_set}_results.pkl", 'wb'))

    # Save final complete results
    final_data = {
            'error_data': error_data,
            'seqstr_data': seqstr_data,
            'budgets': budgets,
            'n_unitaries_per_budget': n_unitaries_per_budget
        }
    pickle.dump(final_data, open(f"{save_dir}/{gate_set}_results.pkl", 'wb'))
    return final_data




def run_benchmark(args):
    return benchmark_on_random_unitaries(**args)


if __name__ == "__main__":
    n_unitaries = 500
    max_budget = 14
    save_dir = f'./benchmark_results_{n_unitaries}'
    gate_set_cost_3 = "tqshxyz_tequiv_medium_cost_3" #"tqshxyz_tequiv_short_cost_3"
    gate_set_cost_2 = "tqshxyz_tequiv_medium" #"tqshxyz_tequiv_short"
    gate_set_cost_25 = "tqshxyz_tequiv_medium_cost_2.5"
    gate_set_t = "tshxyz_tequiv_medium"


    tasks = [
        dict(
            n_unitaries_per_budget=n_unitaries,
            budgets=np.arange(2, max_budget, 0.5),
            gate_set=gate_set_cost_25,
            save_dir=save_dir,
            costs={"T": 1.0, "sqrtT": 2.5},
        ),
        dict(
            n_unitaries_per_budget=n_unitaries,
            budgets=np.arange(2, max_budget),
            gate_set=gate_set_t,
            save_dir=save_dir,
            costs={"T": 1.0},
        ),
        dict(
            n_unitaries_per_budget=n_unitaries,
            budgets=np.arange(2, max_budget),
            gate_set=gate_set_cost_2,
            save_dir=save_dir,
            costs={"T": 1.0, "sqrtT": 2.0},
        ),
    ]
    
    with Pool(processes=3) as pool:  # or processes=None for "cpu_count()"
        pool.map(run_benchmark, tasks)