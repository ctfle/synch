import random

import numpy as np
from numpy.typing import NDArray
import pytest
from trasyn.synthesis import Sythesiser, BudgetPartitioner
from trasyn.utils import random_unitary_2x2, seq2mat, distance


def unitary_from_budget(t_budget: int, q_budget: int, total_len: int) -> str:
    """ Generate a str with t_budget-many t and q_budget-many q"""
    if q_budget + t_budget > total_len:
        raise ValueError("n_q + n_t cannot exceed total_len")

    # Fixed part
    seq = ['q'] * q_budget + ['t'] * t_budget

    # Remaining positions filled with other letters
    others = ['h', 's', 'x', 'y', 'z']
    remaining = total_len - len(seq)
    seq += random.choices(others, k=remaining)

    # Shuffle in-place to randomize order
    random.shuffle(seq)
    return ''.join(seq)

class TestSynthesis():

    @pytest.mark.parametrize("load_dir, costs", [
        ("../assets/tqshxyz_tequiv_medium_cost_2.5", {"T": 1.0, "sqrtT": 2.5}),
        ("../assets/tshxyz_tequiv_medium", {"T": 1.0}),
        ("../assets/tqshxyz_tequiv_medium_cost_2", {"T": 1.0, "sqrtT": 2})
    ])
    @pytest.mark.parametrize("budget", [
        2, 3, 4, 5, 6
    ])
    def test_synthesis(self, budget, load_dir, costs):
        """
        Test that the synthesis actually produces unitaries that are as close to the target
        unitary as the synthesis proclaims.
        """
        partitioner = BudgetPartitioner(
            max_partition_value=5, total_non_clifford_budget=budget,
            costs=costs
        )
        syn = Sythesiser(partitioner=partitioner, load_dir=load_dir)

        for i in range(5):
            target_unitary = random_unitary_2x2()
            result = syn.sample_and_synthesize(target_unitary, verbose=False)
            assert np.allclose(distance(target_unitary, seq2mat(result.seqstr)), result.error)

    @pytest.mark.parametrize("load_dir, costs", [
        ("../assets/tqshxyz_tequiv_medium_cost_2.5", {"T": 1.0, "sqrtT": 2.5}),
        ("../assets/tshxyz_tequiv_medium", {"T": 1.0}),
        ("../assets/tqshxyz_tequiv_medium_cost_2", {"T": 1.0, "sqrtT": 2})
    ])
    @pytest.mark.parametrize("budget", [
        2, 3, 4, 5
    ])
    def test_reverse_engineer_synthesis(self, budget, load_dir, costs):
        """
        This tests creates a unitary using a sequence of t and clifford gates and uses the
        snythesiser to synthesis it. Since we know that the sequence can be created exactly using
        t + Clifford, the synthesiser should find it (or a variation using duplicates). Thus, error
        should be very small.
        """

        target_unitary_seq = unitary_from_budget(t_budget=budget, q_budget=0, total_len=10)
        target_unitary = seq2mat(target_unitary_seq)
        partitioner = BudgetPartitioner(
            max_partition_value=5, total_non_clifford_budget=budget,
            costs=costs
        )
        syn = Sythesiser(partitioner=partitioner, load_dir=load_dir)

        result = syn.sample_and_synthesize(target_unitary, verbose=False)
        np.allclose(distance(target_unitary, seq2mat(result.seqstr)), result.error)
        assert np.allclose(result.error, 0.0, atol=1e-7)




