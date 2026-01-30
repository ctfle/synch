import numpy as np
import pytest
from trasyn.synthesis import Sythesiser, BudgetPartitioner
from trasyn.utils import random_unitary_2x2, seq2mat, distance



class TestSynthesis():
    
    @pytest.mark.parametrize("load_dir, costs", [
        ("../trasyn/assets/tqshxyz_tequiv_medium_cost_2.5", {"T": 1.0, "sqrtT": 2.5}),
        ("../trasyn/assets/tshxyz_tequiv_medium", {"T": 1.0}),
        ("../trasyn/assets/tqshxyz_tequiv_medium_cost_2", {"T": 1.0, "sqrtT": 2})
    ])
    @pytest.mark.parametrize("budget", [
        2, 3, 4, 5, 6
    ])
    def test_synthesis(self, budget, load_dir, costs):

        partitioner = BudgetPartitioner(
            max_partition_value=5, total_non_clifford_budget=budget,
            costs=costs
        )
        syn = Sythesiser(partitioner=partitioner, load_dir=load_dir)

        for i in range(1):
            target_unitary = random_unitary_2x2()
            result = syn.sample_and_synthesize(target_unitary, verbose=False)
            assert np.allclose(distance(target_unitary, seq2mat(result.seqstr)), result.error)
            

            
        
            
        

        